import time
from concurrent import futures
import grpc
import spidev

import rip_service_pb2_grpc
from rip_service_pb2 import RipResponse
import math
spi = spidev.SpiDev()
spi.open(0, 0)
spi.max_speed_hz = 1000000 # NOTE
UNIT_TIME = 0.06
PI = math.pi

class RotaryDoubleInvertedPendulum:
    def __init__(self):
        self.step_idx = 0
        self.last_step_call = 0.0

        spi.xfer2([
            0x40, 0x00, 0x00
        ])
        spi.xfer2([
            0x40, 0x00, 0x10, 0x00, 0x00
        ])
        spi.xfer2([
            0x40, 0x00, 0x00
        ])
        spi.xfer2([
            0x40, 0x00, 0x10, 0x00, 0x00
        ])
        spi.xfer2([
            0x40, 0x00, 0x00
        ])
        print("INITIATION!!!!")
        arm_angle, arm_velocity, link_1_angle, link_1_velocity, link_2_angle, link_2_velocity = self.calculate_state()
        self.print_state(arm_angle, arm_velocity, link_1_angle, link_1_velocity, link_2_angle, link_2_velocity)
        #
        # t = 0
        # while True:
        #     action = 200 * math.sin(2 * 0.1 * math.pi * t)
        #     self.apply_action(int(action))
        #     t += 0.008
        #     time.sleep(0.008)
        #     print(t, int(action))

    def calculate_state(self):
        #data = spi.xfer2([128 if i == 0 else i for i in range(21)])

        # data = spi.xfer2([128, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
        data = spi.xfer3([
            0xC0,
            0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00
            ])


        #t = float((data[1] << 24) + (data[2] << 16) + (data[3] << 8) + data[4])
        arm_angle = float((data[5] << 24) + (data[6] << 16) + (data[7] << 8) + data[8])
        arm_vel = float((data[9] << 24) + (data[10] << 16) + (data[11] << 8) + data[12])
        link_1_angle = float((data[13] << 24) + (data[14] << 16) + (data[15] << 8) + data[16])
        link_1_vel = float((data[17] << 24) + (data[18] << 16) + (data[19] << 8) + data[20])
        link_2_angle = float((data[21] << 24) + (data[22] << 16) + (data[23] << 8) + data[24])
        link_2_vel = float((data[25] << 24) + (data[26] << 16) + (data[27] << 8) + data[28])

        #t = t / 1000
        arm_angle = -(4294967296 - arm_angle) / 100 if arm_angle > 4000000000 else arm_angle / 100
        arm_vel = -(4294967296 - arm_vel) / 100 if arm_vel > 4000000000 else arm_vel / 100
        link_1_angle = -(4294967296 - link_1_angle) / 100 if link_1_angle > 4000000000 else link_1_angle / 100
        link_1_vel = -(4294967296 - link_1_vel) / 100 if link_1_vel > 4000000000 else link_1_vel / 100
        link_2_angle = -(4294967296 - link_2_angle) / 100 if link_2_angle > 4000000000 else link_2_angle / 100
        link_2_vel = -(4294967296 - link_2_vel) / 100 if link_2_vel > 4000000000 else link_2_vel / 100

        return arm_angle, arm_vel, link_1_angle, link_1_vel, link_2_angle, link_2_vel

    def calculate_action(self, motor_power):

        motor_power_hex = hex(motor_power)
        motor_power_str = str(motor_power_hex)
        if len(motor_power_str) < 5:
            front_hex = 0x00
            back_hex = int(motor_power_str, 16)
        else:
            front_hex_str = '0x0' + motor_power_str[2]
            back_hex_str = '0x' + motor_power_str[3:]
            front_hex = int(front_hex_str, 16)
            back_hex = int(back_hex_str, 16)

        return front_hex, back_hex

    def apply_action(self, motor_power):

        if motor_power > 0:
            action_1, action_2 = self.calculate_action(motor_power)
            spi.xfer2([0x40, 0x00, 0x02, action_1, action_2])
        else:
            motor_power = -motor_power
            action_1, action_2 = self.calculate_action(motor_power)
            spi.xfer2([0x40, 0x00, 0x03, action_1, action_2])


        # print("spi write elapsed time : {0:10.8f} \n\n".format(time.time() - last_time))

    def reset(self, rip_request, context):
        arm_angle, arm_velocity, link_1_angle, link_1_velocity, link_2_angle, link_2_velocity = self.calculate_state()

        # self.print_state(arm_angle, arm_velocity, link_angle, link_velocity)

        self.last_step_call = time.time()

        return RipResponse(
            message='OK',
            arm_angle=arm_angle, arm_velocity=arm_velocity,
            link_1_angle=link_1_angle, link_1_velocity=link_1_velocity,
            link_2_angle=link_2_angle, link_2_velocity=link_2_velocity
        )

    def step(self, rip_request, context):
        motor_power = int(rip_request.value)

        # current_step_call = time.time()
        # elapsed_time = current_step_call - self.last_step_call
        # print(self.step_idx, elapsed_time, motor_power)
        # self.last_step_call = time.time()

        self.apply_action(motor_power)

        arm_angle, arm_velocity, link_1_angle, link_1_velocity, link_2_angle, link_2_velocity = self.calculate_state()

        # self.print_state(arm_angle, arm_velocity, link_1_angle, link_1_velocity, link_2_angle, link_2_velocity)

        self.step_idx += 1

        return RipResponse(
            message='STEP',
            arm_angle=arm_angle, arm_velocity=arm_velocity,
            link_1_angle=link_1_angle, link_1_velocity=link_1_velocity,
            link_2_angle=link_2_angle, link_2_velocity=link_2_velocity
        )

    def terminate(self, rip_request, context):
        spi.xfer2([0x40, 0x00, 0x10, 0x00, 0x00])
        spi.xfer2([0x40, 0x00, 0x01])

        return RipResponse(message='OK')

    def print_state(self, arm_angle, arm_velocity, link_1_angle, link_1_velocity, link_2_angle, link_2_velocity):
        print("arm angle :", arm_angle)
        print("arm vel :", arm_velocity)
        print("link 1 angle :", link_1_angle)
        print("link 1 vel :", link_1_velocity)
        print("link 2 angle :", link_2_angle)
        print("link 2 vel :", link_2_velocity)

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

                arm_angle, arm_velocity, link_1_angle, link_1_velocity, link_2_angle, link_2_velocity = self.calculate_state()

                if 0.0 <= abs(link_1_angle) <= PI * 11.0 / 180.0 and 0.0 <= abs(link_2_angle) <= PI * 11.0 / 180.0:
                    break

                # angular_variation = (pendulum_radian - last_pendulum_radian)
                # # angular variation filtering
                # if angular_variation > 2.5:
                #     angular_variation -= math.pi * 2
                # elif angular_variation < -2.5:
                #     angular_variation += math.pi * 2
                #
                # pendulum_angular_velocity = angular_variation / UNIT_TIME

                last_pendulum_radian_1 = link_1_angle
                last_pendulum_radian_2 = link_2_angle

                voltage = 80.0#48.65 # 49.215

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

                self.__set_motor_command(motorPWM, "blue")

        print("\n***** Swing Up complete!!! *****")

if __name__ == "__main__":
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    rip_service_pb2_grpc.add_RDIPServicer_to_server(RotaryDoubleInvertedPendulum(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    server.wait_for_termination()
