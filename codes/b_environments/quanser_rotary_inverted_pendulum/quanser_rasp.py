import spidev
from concurrent import futures
import time
import math

import grpc
import quanser_service_pb2_grpc
from quanser_service_pb2 import QuanserResponse

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

        self.initial_state()
        self.motor_limit = False

    def step(self, QuanserRequest):
        global last_time

        self.color = "green"

        self_servo.motor_command = int(QuanserRequest.value)

        self_servo.set_motor_command()
        motor_radian, motor_velocity, pendulum_radian, pendulum_velocity, step_id = self_servo.read_and_pub()
        self.limit_check()
        return QuanserResponse(
            message="STEP",
            motor_radian=motor_radian, motor_velocity=motor_velocity,
            pendulum_radian=pendulum_radian, pendulum_velocity=pendulum_velocity,
            is_motor_limit=self.motor_limit
        )
    
    def reset(self, QuanserRequest):
        self.motor_limit = False
        motor_radian, motor_velocity, pendulum_radian, pendulum_velocity, step_id, _  = self_servo.set_wait()
        return QuanserResponse(
            message="RESET",
            motor_radian=motor_radian, motor_velocity=motor_velocity,
            pendulum_radian=pendulum_radian, pendulum_velocity=pendulum_velocity
        )

    def pendulum_reset(self, QuanserRequest):
        self.color = "yellow"

        self.spi.xfer2([
            0x01,
            0x00,
            0b01011111,
            0x03, 0xe7, 0x00, 0x00, 0x03, 0xe7,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00
        ])
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
        
        print("__motor_command_split : ", motor_command, type(motor_command)) 

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
        print("motor_command : ",motor_command)
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

    def limit_check(self):
        motor_radian, _ = self.__set_motor_command(self.motor_command)

        if motor_radian > PI / 1.4 or motor_radian < -PI / 2:
            self.motor_limit = True
            self.protection()
            # print("<<=== pub reset complete")

    def protection(self):
        is_protect = True
        while is_protect:
            start_motor_radian, _ = self.read_data()

            time.sleep(UNIT_TIME)

            motor_command = 150 if start_motor_radian > 0 else -150
            end_motor_radian, _ = self.__set_motor_command(motor_command)

            motor_velocity = end_motor_radian - start_motor_radian
            # print("limit m_v:", motor_velocity)

            motor_command = 0
            if end_motor_radian > 0:
                if motor_velocity > 0:
                    motor_command = MOTOR_PROTECTION_VOLTAGE
            else:
                if motor_velocity < 0:
                    motor_command = -MOTOR_PROTECTION_VOLTAGE

            protect_motor_radian, _ = self.__set_motor_command(motor_command)

            time.sleep(UNIT_TIME)

            if abs(protect_motor_radian) < 8 * PI / 18:
                is_protect = False

    # read radian and if step_id is changed, publish to env.
    def read_and_pub(self):
        motor_radian, pendulum_radian = self.__set_motor_command(int(self.motor_command))

        motor_velocity = (motor_radian - self.last_motor_radian) / (UNIT_TIME * 5)
        pendulum_velocity = (pendulum_radian - self.last_pendulum_radian) / (UNIT_TIME * 5)

        self.last_motor_radian = motor_radian
        self.last_pendulum_radian = pendulum_radian
        
        return motor_radian, motor_velocity, pendulum_radian, pendulum_velocity, float(self.step_id)

    # set motor command to last subscribe command.
    def set_motor_command(self):
        self.is_action = True
        # print(datetime.utcnow().strftime('%S.%f')[:-1], self.motor_command, flush = True)
        self.__set_motor_command(self.motor_command)

    def set_wait(self):
        self.is_action = False
        self.color = "white"
        self.__set_motor_command(0)

        motor_radian, pendulum_radian = self.read_data()

        if self.pendulum_count == 11:
            self.pendulum_count = 0
        self.pendulum_count += 1
        return motor_radian, 0, pendulum_radian, 0 ,int(self.step_id), self.pendulum_count


if __name__ == "__main__":
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    quanser_service_pb2_grpc.add_QuanserRIPServicer_to_server(QubeServo2(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    server.wait_for_termination()