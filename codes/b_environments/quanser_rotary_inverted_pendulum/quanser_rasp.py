import paho.mqtt.client as mqtt
import threading
import random
import spidev
from concurrent import futures
import time
import math
import numpy as np
from datetime import datetime
import json
import datetime

import grpc
from codes.b_environments.quanser_rotary_inverted_pendulum import quanser_service_pb2_grpc
from codes.b_environments.quanser_rotary_inverted_pendulum.quanser_service_pb2 import QuanserStateResponse

self_servo = None


PI = math.pi
MOTOR_PROTECTION_VOLTAGE = 500
UNIT_TIME = 1 / 1000

last_time = time.time()


class QubeServo2:
    def __init__(self):
        # self.log_f = "/home/pi/spi/log_file.txt"
        # self.sub_t_list = []
        # self.pub_t_list = []

        global self_servo
        self_servo = self
        self.pub_id_for_edgex = 0
        self.pendulum_count = 0
        self.spi = spidev.SpiDev()
        self.spi.open(0, 0)
        self.spi.mode = 0b10
        self.spi.max_speed_hz = 1000000

        self.step = 0
        self.last_step = 0
        self.motor_command = 0
        self.is_swing_up = True
        self.is_reset = False
        self.last_motor_radian = 0
        self.last_pendulum_radian = 0
        self.is_action = False

        self.last_sub_time = 0.0
        self.last_pub_time = 0.0

        self.reset()
        self.motor_limit = False
        self.reset_complete = False

    def step(self, QuanserStepRequest):
        global last_time

        self.motor_limit = False
        self.reset_complete = False

        self_servo.motor_command = QuanserStepRequest.value
        info = QuanserStepRequest.info
        self_servo.step = QuanserStepRequest.step

        if info == "swingup":
            self_servo.is_swing_up = True
        elif info == "balance":
            current_time = time.time()
            elapsed = current_time - last_time
            print(elapsed * 1000)
            last_time = current_time
            self_servo.is_swing_up = False
            self_servo.set_motor_command()
            self_servo.read_and_pub()

            motor_radian, motor_velocity, pendulum_radian, pendulum_velocity, step, _ = self_servo.read_and_pub()
            self.limit_check()
            return QuanserStateResponse(
                message="OK",
                motor_radian=motor_radian, motor_velocity=motor_velocity,
                pendulum_radian=pendulum_radian, pendulum_velocity=pendulum_velocity,
                step=step, is_motor_limit=self.motor_limit, reset_complete=self.reset_complete
            )
        elif info == "wait":
            motor_radian, motor_velocity, pendulum_radian, pendulum_velocity, step, _ = self_servo.set_wait()
            return QuanserStateResponse(
                message = "OK",
                motor_radian=motor_radian, motor_velocity=motor_velocity,
                pendulum_radian=pendulum_radian, pendulum_velocity=pendulum_velocity,
                step=step, is_motor_limit=self.motor_limit, reset_complete=self.reset_complete
            )
        elif info == "pendulum_reset":
            print("pendulum_reset")
            self_servo.pendulum_reset(self_servo.step)

        elif info == "reset":
            print("reset")
            self.manual_swing_up()
            self.manual_balance()

    def reset(self):
        self.spi.xfer2([
            0x01,
            0x00,
            0b01111111,
            0x03, 0xe7, 0x00, 0x00, 0x03, 0xe7,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00
        ])
        data = self.spi.xfer2([
            0x01,
            0x00,
            0b00011111,
            0x03, 0xe7, 0x00, 0x00, 0x03, 0xe7,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00
        ])

        _, motor_radian, pendulum_radian = self.__data_conversion(data)

    def __data_conversion(self, data):
        # Devoid ID
        device_id = ((data[0] & 0xff) << 8) | (data[1] & 0xff)

        # Motor Encoder Counts
        encoder0 = ((data[2] & 0xff) << 16) | ((data[3] & 0xff) << 8) | (data[4] & 0xff)
        if encoder0 & 0x00800000:
            encoder0 = encoder0 | 0xFF000000
            encoder0 = (0x100000000 - encoder0) * (-1)

        # convert the arm encoder counts to angle theta in radians
        motor_position = encoder0 * (-2.0 * PI / 2048.0)

        # Pendulum Encoder Counts
        encoder1 = ((data[5] & 0xff) << 16) | ((data[6] & 0xff) << 8) | (data[7] & 0xff)
        if encoder1 & 0x00800000:
            encoder1 = encoder1 | 0xFF000000
            encoder1 = (0x100000000 - encoder1) * (-1)

        # wrap the pendulum encoder counts when the pendulum is rotated more than 360 degrees
        encoder1 = encoder1 % 2048
        if encoder1 < 0:
            encoder1 += 2048

        # convert the arm encoder counts to angle theta in radians
        pendulum_angle = encoder1 * (2.0 * PI / 2048.0) - PI

        return device_id, motor_position, pendulum_angle

    def __motor_command_split(self, motor_command):
        # to signed
        if motor_command & 0x0400:
            motor_command = motor_command | 0xfc00

        # add amplifier bit
        motor_command = (motor_command & 0x7fff) | 0x8000

        # separate into 2 bytes
        motor_command_h = (motor_command & 0xff00) >> 8
        motor_command_l = (motor_command & 0xff)
        return motor_command_h, motor_command_l

    def read_data(self):
        data = self.spi.xfer2([
            0x01,
            0x00,
            0x1f,
            0x00, 0xff, 0x00, 0xff, 0x00, 0xff,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00
        ])
        _, motor_radian, pendulum_radian = self.__data_conversion(data)
        return motor_radian, pendulum_radian

    def __set_motor_command(self, motor_command, color):
        if color == "red":
            red_h, red_l, green_h, green_l, blue_h, blue_l = 0x03, 0xe7, 0x00, 0x00, 0x00, 0x00
        elif color == "green":
            red_h, red_l, green_h, green_l, blue_h, blue_l = 0x00, 0x00, 0x03, 0xe7, 0x00, 0x00
        elif color == "blue":
            red_h, red_l, green_h, green_l, blue_h, blue_l = 0x00, 0x00, 0x00, 0x00, 0x03, 0xe7
        elif color == "cyan":
            red_h, red_l, green_h, green_l, blue_h, blue_l = 0x00, 0x00, 0x03, 0xe7, 0x03, 0xe7
        elif color == "magenta":
            red_h, red_l, green_h, green_l, blue_h, blue_l = 0x03, 0xe7, 0x00, 0x00, 0x03, 0xe7
        elif color == "yellow":
            red_h, red_l, green_h, green_l, blue_h, blue_l = 0x03, 0xe7, 0x03, 0xe7, 0x00, 0x00
        elif color == "white":
            red_h, red_l, green_h, green_l, blue_h, blue_l = 0x03, 0xe7, 0x03, 0xe7, 0x03, 0xe7
        else:
            red_h, red_l, green_h, green_l, blue_h, blue_l = 0x00, 0x00, 0x00, 0x00, 0x00, 0x00

        motor_command_h, motor_command_l = self.__motor_command_split(motor_command)

        data = self.spi.xfer2([
            0x01,
            0x00,
            0x1f,
            red_h, red_l, green_h, green_l, blue_h, blue_l,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            motor_command_h, motor_command_l
        ])
        _, motor_radian, pendulum_radian = self.__data_conversion(data)

        return motor_radian, pendulum_radian

    def pendulum_reset(self, step):
        self.spi.xfer2([
            0x01,
            0x00,
            0b01011111,
            0x03, 0xe7, 0x00, 0x00, 0x03, 0xe7,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00
        ])

        print("***** Pendulum Reset Complete!!! step : {} ***** ".format(step))



    def limit_check(self):
        motor_radian, _ = self.__set_motor_command(self.motor_command, "green")

        if motor_radian > PI / 1.4 or motor_radian < -PI / 2:
            self.motor_limit = True
            self.protection()
            self.reset_complete = True
            # print("<<=== pub reset complete")
        else:
            return


    def protection(self):
        is_protect = True
        while is_protect:
            start_motor_radian, _ = self.read_data()

            time.sleep(UNIT_TIME)

            motor_command = 150 if start_motor_radian > 0 else -150
            end_motor_radian, _ = self.__set_motor_command(motor_command, "red")

            motor_velocity = end_motor_radian - start_motor_radian
            # print("limit m_v:", motor_velocity)

            motor_command = 0
            if end_motor_radian > 0:
                if motor_velocity > 0:
                    motor_command = MOTOR_PROTECTION_VOLTAGE
            else:
                if motor_velocity < 0:
                    motor_command = -MOTOR_PROTECTION_VOLTAGE

            protect_motor_radian, _ = self.__set_motor_command(motor_command, "red")

            time.sleep(UNIT_TIME)

            if abs(protect_motor_radian) < 8 * PI / 18:
                is_protect = False

    def manual_swing_up(self):
        print("\n***** Swing Up Start!!! *****")

        previous_time = time.perf_counter()
        last_pendulum_radian = 0
        motor_PWM = 0

        while True:
            # if the difference between the current time and the last time an SPI transaction
            # occurred is greater than the sample time, start a new SPI transaction
            current_time = time.perf_counter()
            if current_time - previous_time >= UNIT_TIME:
                # print("|| Time difference: {0} s ||".format(current_time - previous_time))

                previous_time = current_time

                motor_radian, pendulum_radian = self.read_data()

                if 0.0 <= abs(pendulum_radian) <= PI * 11.0 / 180.0:
                    break

                angular_variation = (pendulum_radian - last_pendulum_radian)
                # angular variation filtering
                if angular_variation > 2.5:
                    angular_variation -= math.pi * 2
                elif angular_variation < -2.5:
                    angular_variation += math.pi * 2

                pendulum_angular_velocity = angular_variation / UNIT_TIME

                last_pendulum_radian = pendulum_radian

                voltage = 80.0  # 48.65 # 49.215

                if abs(pendulum_angular_velocity) > 25:
                    voltage /= int(10 * np.log(abs(pendulum_angular_velocity)))

                if PI >= abs(pendulum_radian) >= PI * 90.0 / 180.0:
                    if pendulum_radian >= 0:
                        pendulum_radian = math.pi - pendulum_radian
                    else:
                        pendulum_radian = -math.pi + abs(pendulum_radian)

                    if pendulum_angular_velocity == 0:
                        if random.random() < 0.5:
                            motor_PWM = int(-2 * math.cos(pendulum_radian) * voltage)
                        else:
                            motor_PWM = int(2 * math.cos(pendulum_radian) * voltage)
                    elif pendulum_angular_velocity < 0:
                        motor_PWM = int(-2 * math.cos(pendulum_radian) * voltage)
                    else:
                        motor_PWM = int(2 * math.cos(pendulum_radian) * voltage)

                self.__set_motor_command(motor_PWM, "blue")

        print("\n***** Swing Up complete!!! *****")

    def manual_balance(self):
        theta_n_k1 = 0.0
        theta_dot_k1 = 0.0
        alpha_n_k1 = 0.0
        alpha_dot_k1 = 0.0

        kp_theta = 2.0
        kd_theta = -2.0
        kp_alpha = -30.0
        kd_alpha = 2.5

        previous_time = time.perf_counter()

        count = 0

        # time
        while count < 1500 / 5:
            # if the difference between the current time and the last time an SPI transaction
            # occurred is greater than the sample time, start a new SPI transaction
            current_time = time.perf_counter()
            if current_time - previous_time >= UNIT_TIME * 5:
                # print("|| Time difference: {0} s ||".format(current_time - previous_time))

                previous_time = current_time

                # LED Blue
                theta, alpha = self.read_data()

                # if the pendulum is within +/-30 degrees of upright, enable balance control
                if abs(alpha) <= (30.0 * math.pi / 180.0):
                    # transfer function = 50s/(s+50)
                    # z-transform at 1ms = (50z - 50)/(z-0.9512)
                    theta_n = -theta
                    theta_dot = (50.0 * theta_n) - (50.0 * theta_n_k1) + (0.7612 * theta_dot_k1)
                    theta_n_k1 = theta_n
                    theta_dot_k1 = theta_dot

                    # transfer function = 50s/(s+50)
                    # z-transform at 1ms = (50z - 50)/(z-0.9512)
                    alpha_n = -alpha
                    alpha_dot = (50.0 * alpha_n) - (50.0 * alpha_n_k1) + (0.7612 * alpha_dot_k1)
                    alpha_n_k1 = alpha_n
                    alpha_dot_k1 = alpha_dot

                    # multiply by proportional and derivative gains
                    motor_voltage = (theta * kp_theta) + (theta_dot * kd_theta) + (alpha * kp_alpha) + (
                            alpha_dot * kd_alpha)

                    # set the saturation limit to +/- 15V
                    if motor_voltage > 15.0:
                        motor_voltage = 15.0
                    elif motor_voltage < -15.0:
                        motor_voltage = -15.0

                    # invert for positive CCW
                    motor_voltage = -motor_voltage

                    # convert the analog value to the PWM duty cycle that will produce the same average voltage
                    motor_PWM = int(motor_voltage * (625.0 / 15.0))
                    if motor_PWM > 280:
                        motor_PWM = 280
                    elif motor_PWM < -280:
                        motor_PWM = -280

                    # print(motor_PWM)

                    self.__set_motor_command(motor_PWM, "green")

                    count += 1

                else:
                    self.read_data()
                    break
        self.last_motor_radian = theta
        self.last_pendulum_radian = alpha
        self.is_reset = False

    # read radian and if step is changed, publish to env.
    def read_and_pub(self):
        if self.step != self.last_step and self.is_action:
            motor_radian, pendulum_radian = self.__set_motor_command(self.motor_command, "green")

            motor_velocity = (motor_radian - self.last_motor_radian) / (UNIT_TIME * 5)
            pendulum_velocity = (pendulum_radian - self.last_pendulum_radian) / (UNIT_TIME * 5)

            self.last_motor_radian = motor_radian
            self.last_pendulum_radian = pendulum_radian

            self.is_action = False
            self.last_step = self.step

            # self.pub.publish(
            #    topic = MQTT_PUB_TO_ENV,
            #    payload = "{0}|{1}|{2}|{3}|{4}".format(
            #        motor_radian, motor_velocity, pendulum_radian, pendulum_velocity, self.step),
            #    qos = 0
            # )
            # current_time = float(datetime.utcnow().strftime('%S.%f')[:-1])
            # with open(self.log_f, 'a') as log:
            #     t_d = current_time - self.last_pub_time
            #     log.write(" <== pub time : {:f}".format(t_d) + '\n')
            #     if not t_d > 1.0:
            #         self.pub_t_list.append(t_d)
            # self.last_pub_time = current_time
            if self.pub_id_for_edgex == 11:
                self.pub_id_for_edgex = 0
            # print("[INFO] motor_radian- ", motor_radian)
            # print("[INFO] motor_velocity- ", motor_velocity)
            # print("[INFO] pendulum_radian- ", pendulum_radian)
            # print("[INFO] pendulum_velocity- ", pendulum_velocity)
            self.pub_id_for_edgex = self.pub_id_for_edgex + 1
            return motor_radian, motor_velocity, pendulum_radian, pendulum_velocity, int(self.step), self.pub_id_for_edgex

    # set motor command to last subscribe command.
    def set_motor_command(self):
        self.is_action = True
        color = "blue" if self.is_swing_up else "green"
        # print(datetime.utcnow().strftime('%S.%f')[:-1], self.motor_command, flush = True)
        self.__set_motor_command(self.motor_command, color)

    def set_wait(self):
        self.is_action = False
        color = "white"
        self.__set_motor_command(0, color)

        motor_radian, pendulum_radian = self.read_data()

        # self.pub.publish(
        #    topic = MQTT_PUB_TO_ENV,
        #    payload = "{0}|{1}|{2}|{3}|{4}".format(
        #        motor_radian, 0, pendulum_radian, 0, self.step),
        #    qos = 0
        # )

        if self.pendulum_count == 11:
            self.pendulum_count = 0
        self.pendulum_count += 1
        return motor_radian, 0, pendulum_radian, 0 ,int(self.step), self.pendulum_count


if __name__ == "__main__":
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    quanser_service_pb2_grpc.add_QuanserRIPServicer_to_server(QubeServo2(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    server.wait_for_termination()