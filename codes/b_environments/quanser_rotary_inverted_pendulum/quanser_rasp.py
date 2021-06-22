import spidev
from concurrent import futures
import time
import math
import numpy as np
import grpc
import quanser_service_pb2_grpc
from quanser_service_pb2 import QuanserResponse
import random

self_servo = None


PI = math.pi
MOTOR_PROTECTION_VOLTAGE = 500
UNIT_TIME = 1 / 1000


class QubeServo2:
    def __init__(self):
        # self.log_f = "/home/pi/spi/log_file.txt"
        # self.sub_t_list = []
        # self.pub_t_list = []

        global self_servo
        self_servo = self
        self.step_id_for_edgex = 0
        self.pendulum_count = 0
        self.spi = spidev.SpiDev()
        self.spi.open(0, 0)
        self.spi.mode = 0b10
        self.spi.max_speed_hz = 1000000

        self.step_id = 0
        self.motor_command = 0
        self.is_reset = False
        self.color = None
        self.last_motor_radian = 0
        self.last_pendulum_radian = 0

        self.motor_limit = False
        
        self.last_time = 0.0
        
        self.pendulum_reset(0,None)

    def step(self, QuanserRequest, context):
        previous_time = time.time()

        self.color = "green"

        self_servo.motor_command = int(QuanserRequest.value)

        self_servo.set_motor_command()
        motor_radian, motor_velocity, pendulum_radian, pendulum_velocity, step_id = self_servo.read_and_pub()
        # print("motor radian :", motor_radian, "pendulum radian :", pendulum_radian)
        self.limit_check(motor_velocity)
        #print("rasp Grpc time :", previous_time - time.time())
        
        return QuanserResponse(
            message="STEP",
            motor_radian=motor_radian, motor_velocity=motor_velocity,
            pendulum_radian=pendulum_radian, pendulum_velocity=pendulum_velocity,
            is_motor_limit=self.motor_limit
        )

    def step_sync(self, QuanserRequest, context):
        motor_radian, motor_velocity, pendulum_radian, pendulum_velocity, step_id = self_servo.read_and_pub()

        return QuanserResponse(
            message="STEP_SYNC",
            motor_radian=motor_radian, motor_velocity=motor_velocity,
            pendulum_radian=pendulum_radian, pendulum_velocity=pendulum_velocity,
            is_motor_limit=self.motor_limit
        )
    
    def reset(self, QuanserRequest, context):
        # self.limit_check()
        self.motor_limit = False
        self.motor_command = 0
        motor_radian, motor_velocity, pendulum_radian, pendulum_velocity, step_id = self_servo.read_and_pub()
        self.protection()
        time.sleep(1)
        return QuanserResponse(
            message="RESET",
            motor_radian=motor_radian, motor_velocity=motor_velocity,
            pendulum_radian=pendulum_radian, pendulum_velocity=pendulum_velocity
        )

    def reset_sync(self, QuanserRequest, context):
        motor_radian, motor_velocity, pendulum_radian, pendulum_velocity, step_id = self_servo.read_and_pub()
        return QuanserResponse(
            message="RESET_SYNC",
            motor_radian=motor_radian, motor_velocity=motor_velocity,
            pendulum_radian=pendulum_radian, pendulum_velocity=pendulum_velocity
        )

    def pendulum_reset(self, QuanserRequest, context):
        self.color = "yellow"

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
        print("***** Pendulum Reset Complete!!! ***** ")
        return QuanserResponse(
            message="PENDULUM_RESET"
        )
        
    def initial_state(self):
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
        
        # print("__motor_command_split : ", motor_command, type(motor_command)) 

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

    def __set_motor_command(self, motor_command):
        motor_command = int(motor_command)
        if self.color == "red":
            red_h, red_l, green_h, green_l, blue_h, blue_l = 0x03, 0xe7, 0x00, 0x00, 0x00, 0x00
        elif self.color == "green":
            red_h, red_l, green_h, green_l, blue_h, blue_l = 0x00, 0x00, 0x03, 0xe7, 0x00, 0x00
        elif self.color == "blue":
            red_h, red_l, green_h, green_l, blue_h, blue_l = 0x00, 0x00, 0x00, 0x00, 0x03, 0xe7
        elif self.color == "cyan":
            red_h, red_l, green_h, green_l, blue_h, blue_l = 0x00, 0x00, 0x03, 0xe7, 0x03, 0xe7
        elif self.color == "magenta":
            red_h, red_l, green_h, green_l, blue_h, blue_l = 0x03, 0xe7, 0x00, 0x00, 0x03, 0xe7
        elif self.color == "yellow":
            red_h, red_l, green_h, green_l, blue_h, blue_l = 0x03, 0xe7, 0x03, 0xe7, 0x00, 0x00
        elif self.color == "white":
            red_h, red_l, green_h, green_l, blue_h, blue_l = 0x03, 0xe7, 0x03, 0xe7, 0x03, 0xe7
        else:
            red_h, red_l, green_h, green_l, blue_h, blue_l = 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
        # print("motor_command : ",motor_command)
        motor_command_h, motor_command_l = self.__motor_command_split(motor_command)
        
        
        data = self.spi.xfer2([
            0x01,
            0x00,
            0x1f,
            red_h, red_l, green_h, green_l, blue_h, blue_l,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            motor_command_h, motor_command_l
        ])
        #print("RIP time :", time.time() - previous_time)
        _, motor_radian, pendulum_radian = self.__data_conversion(data)
        return motor_radian, pendulum_radian

    def limit_check(self, motor_velocity):
        motor_radian, _ = self.__set_motor_command(int(self.motor_command))
        # print("motor: ", motor_radian)
        if not self.motor_limit and abs(motor_radian) >= 80 * PI / 180.0:
            self.color = "red"
            for i in range(50):
                if motor_radian >= 0:
                    m = int(abs(self.motor_command) * ((max(motor_velocity, 100) - i) / 100))
                    motor_radian, _ = self.__set_motor_command(m)
                else:
                    m = -int(abs(self.motor_command) * ((max(motor_velocity, 100) - i) / 100))
                    motor_radian, _ = self.__set_motor_command(m)
                #print(m, motor_velocity)
                time.sleep(UNIT_TIME)
            #print()
            self.protection()
            self.motor_limit = True

    def protection(self):
        n_protection = 0
        n_protection_completion = 0
        default_motor_command_ratio = 50
        motor_radian, _ = self.read_data()
        while n_protection_completion < 200:
            time.sleep(UNIT_TIME)
            motor_command = motor_radian * default_motor_command_ratio
            #print("1111111")
            if n_protection % 10 == 0:
                motor_radian, _ = self.__set_motor_command(int(-motor_command))
            else:
                motor_radian, _ = self.__set_motor_command(int(motor_command))
            #time.sleep(UNIT_TIME * 10)
            # print(motor_command)
            if abs(motor_radian) < 10 * PI / 180.0:
                n_protection_completion += 1
            else:
                n_protection_completion = 0
            n_protection += 1
    # read radian and if step_id is changed, publish to env.
    def read_and_pub(self):
        motor_radian, pendulum_radian = self.__set_motor_command(int(self.motor_command))
        current_time = time.time()
        motor_velocity = (motor_radian - self.last_motor_radian) / (current_time - self.last_time)
        pendulum_velocity = (pendulum_radian - self.last_pendulum_radian) / (current_time - self.last_time)

        self.last_motor_radian = motor_radian
        self.last_pendulum_radian = pendulum_radian
        self.last_time = time.time()
        return motor_radian, motor_velocity, pendulum_radian, pendulum_velocity, float(self.step_id)

    # set motor command to last subscribe command.
    def set_motor_command(self):
        self.is_action = True
        # print(datetime.utcnow().strftime('%S.%f')[:-1], self.motor_command, flush = True)
        self.__set_motor_command(int(self.motor_command))

    def set_wait(self):
        self.is_action = False
        self.color = "white"
        self.__set_motor_command(0)

        motor_radian, pendulum_radian = self.read_data()

        return motor_radian, 0, pendulum_radian, 0 ,int(self.step_id)
    
    def manual_swing_up(self):
        print("\n***** Swing Up Start!!! *****")

        previousTime = time.perf_counter()
        last_pendulum_radian = 0
        motorPWM = 0

        while True:
            # if the difference between the current time and the last time an SPI transaction
            # occurred is greater than the sample time, start a new SPI transaction
            currentTime = time.perf_counter()
            if currentTime - previousTime >= UNIT_TIME:
                # print("|| Time difference: {0} s ||".format(currentTime - previousTime))

                previousTime = currentTime

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
                        pendulum_radian = - math.pi + abs(pendulum_radian)

                    if pendulum_angular_velocity == 0:
                        if random.random() < 0.5:
                            motorPWM = int(-2 * math.cos(pendulum_radian) * voltage)
                        else:
                            motorPWM = int(2 * math.cos(pendulum_radian) * voltage)
                    elif pendulum_angular_velocity < 0:
                        motorPWM = int(-2 * math.cos(pendulum_radian) * voltage)
                    else:
                        motorPWM = int(2 * math.cos(pendulum_radian) * voltage)

                self.__set_motor_command(motorPWM)
                
                self.color = "blue"

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

        previousTime = time.perf_counter()

        count = 0

        # time
        while count < 1500 / 5:
            # if the difference between the current time and the last time an SPI transaction
            # occurred is greater than the sample time, start a new SPI transaction
            currentTime = time.perf_counter()
            if currentTime - previousTime >= UNIT_TIME * 5:
                # print("|| Time difference: {0} s ||".format(currentTime - previousTime))

                previousTime = currentTime

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
                    motorVoltage = (theta * kp_theta) + (theta_dot * kd_theta) + (alpha * kp_alpha) + (
                            alpha_dot * kd_alpha)

                    # set the saturation limit to +/- 15V
                    if motorVoltage > 15.0:
                        motorVoltage = 15.0
                    elif motorVoltage < -15.0:
                        motorVoltage = -15.0

                    # invert for positive CCW
                    motorVoltage = -motorVoltage

                    # convert the analog value to the PWM duty cycle that will produce the same average voltage
                    motorPWM = int(motorVoltage * (625.0 / 15.0))
                    if motorPWM > 280:
                        motorPWM = 280
                    elif motorPWM < -280:
                        motorPWM = -280

                    # print(motorPWM)

                    self.__set_motor_command(motorPWM)
                    self.color = "green"

                    count += 1

                else:
                    self.read_data()
                    break
        self.last_motor_radian = theta
        self.last_pendulum_radian = alpha
        self.is_reset = False




if __name__ == "__main__":
    qs = QubeServo2()
    #qs.pendulum_reset(0,None)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    quanser_service_pb2_grpc.add_QuanserRIPServicer_to_server(qs, server)
    server.add_insecure_port('[::]:50051')
    server.start()
    server.wait_for_termination()