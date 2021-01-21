import time

import gym
import numpy as np
import grpc

# MQTT Topic for RIP
from codes.b_environments.quanser_rotary_inverted_pendulum import quanser_service_pb2_grpc
from common.environments.environment import Environment

from codes.b_environments.quanser_rotary_inverted_pendulum.quanser_service_pb2 import QuanserResetRequest, QuanserStepRequest


STATE_SIZE = 4

balance_motor_power_list = [-60, 0, 60]

PUB_ID = 0

RIP_SERVER = '192.168.0.254'


class EnvironmentQuanserRIP(gym.Env):
    def __init__(self):
        super(EnvironmentQuanserRIP, self).__init__()
        self.episode = 0

        self.state_space_shape = (STATE_SIZE,)
        self.action_space_shape = (len(balance_motor_power_list),)

        self.reward = 0

        self.steps = 0
        self.pendulum_radians = []
        self.state = []
        self.current_pendulum_radian = 0
        self.current_pendulum_velocity = 0
        self.current_motor_velocity = 0
        self.previous_time = 0.0

        self.is_swing_up = True
        self.is_state_changed = False
        self.is_motor_limit = False
        self.is_limit_complete = False
        self.is_reset_complete = False

        self.n_states = self.get_n_states()
        self.n_actions = self.get_n_actions()

        self.state_shape = self.get_state_shape()
        self.action_shape = self.get_action_shape()

        self.continuous = False

        self.global_step = 0

        channel = grpc.insecure_channel('{0}:50051'.format(RIP_SERVER))
        self.server_obj = quanser_service_pb2_grpc.QuanserRIPStub(channel)

    def __pendulum_reset(self):
        quanser_response = self.server_obj.step(QuanserStepRequest(value=0, info='pendulum_reset', step=self.global_step))
        if quanser_response.message != "OK":
            raise ValueError()
        # require_response=False

    # RIP Manual Swing & Balance
    def manual_swingup_balance(self):
        quanser_response = self.server_obj.step(QuanserStepRequest(value=0, info='reset', step=self.global_step))
        if quanser_response.message != "OK":
            raise ValueError()

    # for restarting episode
    def wait(self):
        quanser_response = self.server_obj.step(QuanserStepRequest(value=0, info='wait', step=self.global_step))
        if quanser_response.message != "OK":
            raise ValueError()

    def get_n_states(self):
        n_states = 4
        return n_states

    def get_n_actions(self):
        n_actions = 3
        return n_actions

    def get_state_shape(self):
        state_shape = (2,)
        return state_shape

    def get_action_shape(self):
        action_shape = (3,)
        return action_shape

    @property
    def action_meanings(self):
        action_meanings = ["LEFT", "STOP", "RIGHT"]
        return action_meanings

    @property
    def action_meanings(self):
        action_meanings = ["LEFT", "STOP", "RIGHT"]
        return action_meanings

    def reset(self):
        self.steps = 0
        self.pendulum_radians = []
        self.reward = 0
        self.is_motor_limit = False

        wait_time = 1 if self.episode == 0 else 15  # if self.episode % 10 == 0 else 3
        previousTime = time.perf_counter()
        time_done = False

        while not time_done:
            currentTime = time.perf_counter()
            if currentTime - previousTime >= wait_time:
                time_done = True
            time.sleep(0.0001)

        self.__pendulum_reset()
        self.wait()
        self.manual_swingup_balance()
        self.is_motor_limit = False

        self.episode += 1
        self.previous_time = time.perf_counter()

        return np.asarray(self.state)

    def step(self, action):
        motor_power = balance_motor_power_list[int(action)]

        quanser_response = self.server_obj.step(QuanserStepRequest(value=motor_power, info='balance', step=self.global_step))
        if quanser_response.message != "OK":
            raise ValueError()

        motor_radian = quanser_response.motor_radian
        motor_velocity = quanser_response.motor_velocity
        pendulum_radian = quanser_response.pendulum_radian
        pendulum_velocity = quanser_response.pendulum_velocity
        self.is_motor_limit = quanser_response.is_motor_limit
        self.is_limit_complete = quanser_response.reset_complete

        self.state = [pendulum_radian, pendulum_velocity, motor_radian, motor_velocity]
        # self.state = [pendulum_radian, pendulum_velocity]

        self.current_pendulum_radian = pendulum_radian
        self.current_pendulum_velocity = pendulum_velocity
        self.current_motor_velocity = motor_velocity

        pendulum_radian = self.current_pendulum_radian
        pendulum_angular_velocity = self.current_pendulum_velocity

        next_state = np.asarray(self.state)
        self.reward = 1.0
        adjusted_reward = self.reward / 100
        self.steps += 1
        self.pendulum_radians.append(pendulum_radian)
        done, info = self.__isDone()

        if not done:
            while True:
                current_time = time.perf_counter()
                if current_time - self.previous_time >= 6 / 1000:
                    break
        else:
            self.wait()

        self.previous_time = time.perf_counter()

        self.global_step += 1

        return next_state, self.reward, adjusted_reward, done, info

    def __isDone(self):
        info = {}

        def insert_to_info(s):
            info["result"] = s

        if self.steps >= 5000:
            insert_to_info("*** Success ***")
            return True, info
        elif self.is_motor_limit:
            self.reward = 0
            insert_to_info("*** Limit position ***")
            return True, info
        elif abs(self.pendulum_radians[-1]) > 3.14 / 24:
            self.is_fail = True
            self.reward = 0
            insert_to_info("*** Success ***")
            return True, info
        else:
            insert_to_info("")
            return False, info

    def close(self):
        quanser_response = self.server_obj.step(QuanserStepRequest(value=0, info='None', step=self.global_step))
        if quanser_response.message != "OK":
            raise ValueError()