import pickle
import time
from collections import deque
from concurrent import futures
import grpc
import spidev
import threading
import rip_service_pb2_grpc
from rip_service_pb2 import RipResponse
import math
import numpy as np
spi = spidev.SpiDev()
spi.open(0, 0)
spi.max_speed_hz = 5000000 # NOTE


class RotaryDoubleInvertedPendulum:
    def __init__(self):
        self.last_step_call = 0.0
        self.initialize()
        self.step_idx = 0
        self.check_angle_period_steps = 1000000
        self.next_check_angle_step = 0

        # t = 0
        # while True:
        #     action = 200 * math.sin(2 * 0.1 * math.pi * t)
        #     self.apply_action(int(action))
        #     t += 0.008
        #     time.sleep(0.008)
        #     print(t, int(action))
        self.previous_action = 0

        self.count_continuous_fast_pendulum_velocity = 0

    def initialize(self, rip_request=None, context=None):
        spi.xfer2([
            0x40, 0x00, 0x00, 0x00, 0x00
        ])
        spi.xfer2([
            0x40, 0x00, 0x10, 0x00, 0x00
        ])
        spi.xfer2([
            0x40, 0x00, 0x00, 0x00, 0x00
        ])
        spi.xfer2([
            0x40, 0x00, 0x10, 0x00, 0x00
        ])
        spi.xfer2([
            0x40, 0x00, 0x00, 0x00, 0x00
        ])
        print("INITIATION!!!!")
        arm_angle, arm_velocity, link_1_angle, link_1_velocity, link_2_angle, link_2_velocity = self.calculate_state()
        self.print_state(arm_angle, arm_velocity, link_1_angle, link_1_velocity, link_2_angle, link_2_velocity)

        return RipResponse(message='OK')

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
            # action_1 = hex(action_1)
            # action_2 = hex(action_2)
            # print(action_1, action_2)
            spi.xfer2([0x40, 0x00, 0x02, action_1, action_2])

        else:
            motor_power = -motor_power
            action_1, action_2 = self.calculate_action(motor_power)
            # action_1 = hex(action_1)
            # action_2 = hex(action_2)
            # print(action_1,action_2)
            spi.xfer2([0x40, 0x00, 0x03, action_1, action_2])

        # print("spi write elapsed time : {0:10.8f} \n\n".format(time.time() - last_time))

    def reset(self):

        spi.xfer2([0x40, 0x00, 0x01, 0x00, 0x00]) # stop

        if self.step_idx >= self.next_check_angle_step:
            time.sleep(15)
            arm_angle, arm_velocity, link_1_angle, link_1_velocity, link_2_angle, link_2_velocity = self.calculate_state()
            print("[STEP INDEX: {0}] link_1_angle : {1:5.3f}, link_2_angle : {2:5.3f}".format(
                self.step_idx,
                link_1_angle % 360, link_2_angle % 360
            ))
            self.next_check_angle_step += self.check_angle_period_steps
        else:
            time.sleep(3)
            arm_angle, arm_velocity, link_1_angle, link_1_velocity, link_2_angle, link_2_velocity = self.calculate_state()
        # spi.xfer2([0x40, 0x00, 0x10, 0x00, 0x00])

        # self.print_state(arm_angle, arm_velocity, link_angle, link_velocity)

        self.last_step_call = time.time()

    def reset_sync(self, rip_request, context):
        arm_angle, arm_velocity, link_1_angle, link_1_velocity, link_2_angle, link_2_velocity = self.calculate_state()

        # self.print_state(arm_angle, arm_velocity, link_angle, link_velocity)

        return RipResponse(
            message='OK',
            arm_angle=arm_angle, arm_velocity=arm_velocity,
            link_1_angle=link_1_angle, link_1_velocity=link_1_velocity,
            link_2_angle=link_2_angle, link_2_velocity=link_2_velocity
        )

    def step(self, rip_request, context):
        self.step_idx += 1

        motor_power = int(rip_request.value)
        # current_step_call = time.time()
        # elapsed_time = current_step_call - self.last_step_call
        # print(self.step_idx, elapsed_time, motor_power)
        # self.last_step_call = time.time()
        # print(self.step_idx, motor_power)
        if self.previous_action * motor_power < 0:
            spi.xfer2([0x40, 0x00, 0x01, 0x00, 0x00])

        self.apply_action(motor_power)

        arm_angle, arm_velocity, link_1_angle, link_1_velocity, link_2_angle, link_2_velocity = self.calculate_state()

        # self.print_state(arm_angle, arm_velocity, link_1_angle, link_1_velocity, link_2_angle, link_2_velocity)
        self.previous_action = motor_power

        return RipResponse(
            message='OK',
            arm_angle=arm_angle, arm_velocity=arm_velocity,
            link_1_angle=link_1_angle, link_1_velocity=link_1_velocity,
            link_2_angle=link_2_angle, link_2_velocity=link_2_velocity
        )

    def run(self, action):
        motor_power = int(action)
        motor_radian_list = []
        pendulum_radian_list = []
        motor_radian_vel_list = []
        pendulum_radian_vel_list = []

        for i in range(100):
            if i % 2 == 0:
                # motor_power = -motor_power
                self.apply_action(motor_power)
                arm_angle, arm_velocity, link_1_angle, link_1_velocity, link_2_angle, link_2_velocity = self.calculate_state()
            else:
                self.apply_action(-motor_power)
                arm_angle, arm_velocity, link_1_angle, link_1_velocity, link_2_angle, link_2_velocity = self.calculate_state()
            motor_radian_list.append(arm_angle)
            pendulum_radian_list.append(link_1_angle)
            motor_radian_vel_list.append(arm_velocity)
            pendulum_radian_vel_list.append(link_1_velocity)
            # print("motor_radian:", arm_angle)
            # print("pendulum_radian: ", link_1_angle)
            time.sleep(0.1)
        spi.xfer2([0x40, 0x00, 0x01, 0x00, 0x00])
        return motor_radian_list, pendulum_radian_list, motor_radian_vel_list, pendulum_radian_vel_list

    def step_sync(self,rip_request, context):
        motor_power = int(rip_request.value)

        if self.previous_action * motor_power < 0:
            spi.xfer2([0x40, 0x00, 0x01, 0x00, 0x00])

        self.apply_action(motor_power)

        arm_angle, arm_velocity, link_1_angle, link_1_velocity, link_2_angle, link_2_velocity = self.calculate_state()

        return RipResponse(
            message='OK',
            arm_angle=arm_angle, arm_velocity=arm_velocity,
            link_1_angle=link_1_angle, link_1_velocity=link_1_velocity,
            link_2_angle=link_2_angle, link_2_velocity=link_2_velocity
        )

    def force_terminate(self):
        spi.xfer2([0x40, 0x00, 0x01, 0x00, 0x00])
        print("FORCE TERMINATE !!!!!!!!!!!!!!!!!! - count_continuous_fast_pendulum_velocity: {0}".format(
            self.count_continuous_fast_pendulum_velocity
        ))

    def step_test(self):
        t = 0
        action_3 = 0
        while True:
            # if self.step_idx % 2 == 0:
            #     spi.xfer2([0x40, 0x00, 0x01, 0x00, 0x00])
            #     self.action_2 = 200*math.sin(time.time())
            #     self.apply_action(self.action_2)
            # else:
            #     self.apply_action(self.action_2)
            self.action_2 = 200*math.sin(2*math.pi*1.0*t)

            if self.action_2 * action_3 < 0:
                spi.xfer2([0x40, 0x00, 0x01, 0x00, 0x00])

            self.apply_action(int(self.action_2))
            self.step_idx += 1
            action_3 = self.action_2
            t += 0.006
            time.sleep(0.006)

    def terminate(self, rip_request, context):
        spi.xfer2([0x40, 0x00, 0x01, 0x00, 0x00])

        return RipResponse(message='OK')

    def test_terminate(self):
        spi.xfer2([0x40, 0x00, 0x01, 0x00, 0x00])
        return None

    def print_state(self, arm_angle, arm_velocity, link_1_angle, link_1_velocity, link_2_angle, link_2_velocity):
        print("arm angle :", arm_angle)
        print("arm vel :", arm_velocity)
        print("link 1 angle :", link_1_angle)
        print("link 1 vel :", link_1_velocity)
        print("link 2 angle :", link_2_angle)
        print("link 2 vel :", link_2_velocity)

    def savePickle(self, filename, state_list):
        with open(filename, 'wb') as f:
            pickle.dump(state_list, f, pickle.HIGHEST_PROTOCOL)

    def loadPickle(self, filename):
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        return data

    def get_pcc(self, filename_1, filename_2, filename_3):
        data_1 = self.loadPickle(filename_1)
        data_2 = self.loadPickle(filename_2)
        data_3 = self.loadPickle(filename_3)

        len_data_pen_1 = len(data_1)
        len_data_pen_2 = len(data_2)
        len_data_pen_3 = len(data_3)
        # print(len_data_pen_1)
        # print(len_data_pen_2)

        if len_data_pen_1 != len_data_pen_2:
            min_len = min(len_data_pen_1, len_data_pen_2)
            data_pen_1 = data_1[:min_len]
            data_pen_2 = data_2[:min_len]
        # print(len(data_pen_1))
        # print(len(data_pen_2))

        np_data_pen_1 = np.array(data_1)
        np_data_pen_2 = np.array(data_2)
        np_data_pen_3 = np.array(data_3)

        # corr = np.corrcoef(np_data_pen_1, np_data_pen_2)

        corr_1_2 = np.corrcoef(np_data_pen_1, np_data_pen_2)
        corr_1_3 = np.corrcoef(np_data_pen_1, np_data_pen_3)
        corr_2_3 = np.corrcoef(np_data_pen_2, np_data_pen_3)

        corr = np.zeros((3, 3))
        for i in range(3):
            for j in range(3):
                if i == j:
                    corr[i][j] = 1
                elif i == 0 and j == 1:
                    corr[i][j] = corr_1_2[0][1]
                elif i == 0 and j == 2:
                    corr[i][j] = corr_1_3[0][1]
                elif i == 1 and j == 2:
                    corr[i][j] = corr_2_3[0][1]
        print(corr)


def start_grpc(rip, server):
    rip_service_pb2_grpc.add_RDIPServicer_to_server(rip, server)
    server.add_insecure_port('[::]:50051')
    server.start()
    server.wait_for_termination()

def velocity_check(rip, server):
    arm_velocity_num = 0
    link_1_velocity_num = 0
    arm_vel_deque = deque(maxlen=100)
    link_vel_deque = deque(maxlen=100)
    while True:
        arm_angle, arm_velocity, link_1_angle, link_1_velocity, link_2_angle, link_2_velocity = rip.calculate_state()
        arm_vel_deque.append(abs(arm_velocity))
        link_vel_deque.append(abs(link_1_velocity))
        if arm_velocity > 1600:
            arm_velocity_num += 1
        else:
            arm_velocity_num = 0
        if link_1_velocity > 1500:
            link_1_velocity_num += 1
        else:
            link_1_velocity_num = 0
        # arm_vel_deque_mean = sum(arm_vel_deque)/100.0
        # link_vel_deque_mean = sum(link_vel_deque)/100.0
        # if arm_vel_deque_mean > 100 or link_vel_deque_mean > 1800: #5SECOND
        if arm_velocity_num > 1000 or link_1_velocity_num > 1000:
            arm_angle, arm_velocity, link_1_angle, link_1_velocity, link_2_angle, link_2_velocity = rip.calculate_state()
            print("[STEP INDEX: {0}] link_1_angle : {1:5.3f}, link_2_angle : {2:5.3f}".format(
                rip.step_idx,
                link_1_angle % 360, link_2_angle % 360
            ))
            server.stop(grace=None)
            for _ in range(3):
                rip.test_terminate()
            raise Exception("EXCEED VELOCITY!!!")
        time.sleep(0.005)

# if __name__ == "__main__":
#     rip = RotaryDoubleInvertedPendulum()
#     server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
#
#     vel_check = threading.Thread(target=velocity_check, args=(rip, server))
#     vel_check.start()
#     start_grpc(rip, server)
#     # rasp_rip = RotaryDoubleInvertedPendulum()
#     # try:
#     #     rasp_rip.step_test()
#     # except KeyboardInterrupt as e:
#     #     rasp_rip.test_terminate()

if __name__ == "__main__":
    rip = RotaryDoubleInvertedPendulum()
    for j in range(3):
        if j == 0:
            action = 30
        elif j == 1:
            action = 40
        else:
            action = 50
        for i in range(3):
            filename_1 = '[{0}]_arm_angle.pickle'.format(i)
            filename_2 = '[{0}]_link_angle.pickle'.format(i)
            filename_3 = '[{0}]_arm_vel.pickle'.format(i)
            filename_4 = '[{0}]_link_vel.pickle'.format(i)
            rip.reset()
            time.sleep(0.1)

            # measure state
            motor_radian_list, pendulum_radian_list, motor_radian_vel_list, pendulum_radian_vel_list = rip.run(action)

            # save using pickle
            rip.savePickle(filename_1, motor_radian_list)
            rip.savePickle(filename_2, pendulum_radian_list)
            rip.savePickle(filename_3, motor_radian_vel_list)
            rip.savePickle(filename_4, pendulum_radian_vel_list)

            time.sleep(1)
            rip.reset()
            if i != 2:
                rip.initialize()
            time.sleep(10)

        filename_1_motor = '[0]_arm_angle.pickle'
        filename_2_motor = '[1]_arm_angle.pickle'
        filename_3_motor = '[2]_arm_angle.pickle'
        filename_1_pendulum = '[0]_link_angle.pickle'
        filename_2_pendulum = '[1]_link_angle.pickle'
        filename_3_pendulum = '[2]_link_angle.pickle'
        filename_1_motor_vel = '[0]_arm_vel.pickle'
        filename_2_motor_vel = '[1]_arm_vel.pickle'
        filename_3_motor_vel = '[2]_arm_vel.pickle'
        filename_1_pendulum_vel = '[0]_link_vel.pickle'
        filename_2_pendulum_vel = '[1]_link_vel.pickle'
        filename_3_pendulum_vel = '[2]_link_vel.pickle'
        print("action :", action)
        print("corr of arm_angle")
        rip.get_pcc(filename_1_motor, filename_2_motor, filename_3_motor)
        print()
        print("corr of link_angle")
        rip.get_pcc(filename_1_pendulum, filename_2_pendulum, filename_3_pendulum)
        print()
        print("corr of arm_vel")
        rip.get_pcc(filename_1_motor_vel, filename_2_motor_vel, filename_3_motor_vel)
        print()
        print("corr of link_vel")
        rip.get_pcc(filename_1_pendulum_vel, filename_2_pendulum_vel, filename_3_pendulum_vel)







