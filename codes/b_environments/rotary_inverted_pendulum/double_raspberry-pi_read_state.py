import time
from concurrent import futures
import grpc
import spidev
import math

import rip_service_pb2_grpc
from rip_service_pb2 import RipResponse

spi = spidev.SpiDev()
spi.open(0,0)
spi.max_speed_hz = 1000000 # NOTE


class RotaryDoubleInvertedPendulum:
    def __init__(self):
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
        self.print_state()

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
        self.arm_angle = -(4294967296 - arm_angle) / 100 if arm_angle > 4000000000 else arm_angle / 100
        self.arm_vel = -(4294967296 - arm_vel) / 100 if arm_vel > 4000000000 else arm_vel / 100
        self.link_1_angle = -(4294967296 - link_1_angle) / 100 if link_1_angle > 4000000000 else link_1_angle / 100
        self.link_1_vel = -(4294967296 - link_1_vel) / 100 if link_1_vel > 4000000000 else link_1_vel / 100
        self.link_2_angle = -(4294967296 - link_2_angle) / 100 if link_2_angle > 4000000000 else link_2_angle / 100
        self.link_2_vel = -(4294967296 - link_2_vel) / 100 if link_2_vel > 4000000000 else link_2_vel / 100

        return self.arm_angle, self.arm_vel, self.link_1_angle, self.link_1_vel, self.link_2_angle, self.link_2_vel

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
        motor_power = int(motor_power)
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

        return RipResponse(
            message='OK',
            arm_angle=arm_angle, arm_velocity=arm_velocity,
            link_1_angle=link_1_angle, link_1_velocity=link_1_velocity,
            link_2_angle=link_2_angle, link_2_velocity=link_2_velocity
        )

    def step(self, rip_request, context):
        motor_power = int(rip_request.value)
        #print("motor_power :", motor_power)

        self.apply_action(motor_power)

        arm_angle, arm_velocity, link_1_angle, link_1_velocity, link_2_angle, link_2_velocity = self.calculate_state()

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

    def print_state(self):
        print("arm angle :", self.arm_angle)
        print("arm vel :", self.arm_vel)
        print("link 1 angle :", self.link_1_angle)
        print("link 1 vel :", self.link_1_vel)
        print("link 2 angle :", self.link_2_angle)
        print("link 2 vel :", self.link_2_vel)

if __name__ == "__main__":
    double_pendulum = RotaryDoubleInvertedPendulum()
    while True:
        double_pendulum.calculate_state()
        double_pendulum.print_state()
        double_pendulum.apply_action(300*math.sin(time.perf_counter()))
        time.sleep(0.05)
