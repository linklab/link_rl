import time
import math

import gym
from gym import spaces
import numpy as np
import grpc

# MQTT Topic for RIP
from codes.b_environments.quanser_rotary_inverted_pendulum import quanser_service_pb2_grpc
from common.environments.environment import Environment

from codes.b_environments.quanser_rotary_inverted_pendulum.quanser_service_pb2 import QuanserRequest


STATE_SIZE = 4

balance_motor_power_list = [-60., 0., 60.]

RIP_SERVER = '192.168.0.13'


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

        self.count_continuous_uprights = 0
        self.is_upright = False

        self.motor_radian = 0
        self.motor_velocity = 0
        self.pendulum_radian = 0
        self.pendulum_velocity = 0

        self.is_motor_limit = False

        self.n_states = self.get_n_states()
        self.n_actions = self.get_n_actions()

        self.state_shape = self.get_state_shape()
        self.action_shape = self.get_action_shape()
        self.initial_motor_radian = 0.0


        self.continuous = False

        low = np.array([0, 0, 0, 0], dtype=np.float32)
        high = np.array([360, 100, 360, 100], dtype=np.float32)

        self.observation_space = gym.spaces.Box(
            low=low, high=high, dtype=np.float32
        )

        self.action_space = spaces.Discrete(3)

        channel = grpc.insecure_channel('{0}:50051'.format(RIP_SERVER))
        self.server_obj = quanser_service_pb2_grpc.QuanserRIPStub(channel)

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

    def pendulum_reset(self):
        quanser_response = self.server_obj.step(QuanserRequest(value=0.0))
        if quanser_response.message != "PENDULUM_RESET":
            raise ValueError()

    def reset(self):
        self.steps = 0
        self.pendulum_radians = []
        self.reward = 0

        wait_time = 1 if self.episode == 0 else 15  # if self.episode % 10 == 0 else 3
        previousTime = time.perf_counter()
        time_done = False

        quanser_response = self.server_obj.reset(QuanserRequest(value=0.0))
        if quanser_response.message != "RESET":
            raise ValueError()

        self.motor_radian = quanser_response.motor_radian
        self.motor_velocity = quanser_response.motor_velocity
        self.pendulum_radian = quanser_response.pendulum_radian
        self.pendulum_velocity = quanser_response.pendulum_velocity

        self.state = [
            math.cos(self.pendulum_radian),
            math.sin(self.pendulum_radian),
            self.pendulum_velocity,
            math.cos(0),
            math.sin(0),
            self.motor_velocity
        ]

        while not time_done:
            currentTime = time.perf_counter()
            if currentTime - previousTime >= wait_time:
                time_done = True
            time.sleep(0.0001)

        self.initial_motor_radian = self.motor_radian
        self.is_motor_limit = False
        self.episode += 1

        return np.asarray(self.state)

    def step(self, action):
        motor_power = balance_motor_power_list[int(action)]

        #==================== Grpc and use sample time========================================
        previous_time = time.perf_counter()

        quanser_response = self.server_obj.step(QuanserRequest(value=float(motor_power)))
        if quanser_response.message != "STEP":
            raise ValueError()

        while True:
            current_time = time.perf_counter()
            if current_time - previous_time >= 60 / 1000:
                break
            time.sleep(0.0001)
        #=====================================================================================

        self.motor_radian = quanser_response.motor_radian
        self.motor_velocity = quanser_response.motor_velocity
        self.pendulum_radian = quanser_response.pendulum_radian
        self.pendulum_velocity = quanser_response.pendulum_velocity
        self.is_motor_limit = quanser_response.is_motor_limit

        self.state = [
            math.cos(self.pendulum_radian),
            math.sin(self.pendulum_radian),
            self.pendulum_velocity,
            math.cos(self.initial_motor_radian - self.motor_radian),
            math.sin(self.initial_motor_radian - self.motor_radian),
            self.motor_velocity
        ]
        next_state = np.asarray(self.state)


        #=======================reward===============================================================
        self.reward = self.get_reward()
        #=============================================================================================

        self.steps += 1
        done, info = self.__isDone()

        return next_state, self.reward, done, info

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
        elif abs(self.initial_motor_radian - self.motor_radian) > 0.27:
            insert_to_info("*** Relative motor_radian exceed 15***")
            return True, info
        else:
            insert_to_info("")
            return False, info

    def update_current_state(self, ):
        if abs(self.motor_radian) < math.radians(12):
            self.count_continuous_uprights += 1
        else:
            self.count_continuous_uprights = 0

        if self.count_continuous_uprights >= 1:
            self.is_upright = True
        else:
            self.is_upright = False

    def get_reward(self):
        self.update_current_state()

        if self.is_upright:
            position_reward = math.pi - abs(self.pendulum_radian)  # math.pi - math.radians(12) ~ math.pi
        else:
            position_reward = (math.pi - abs(self.pendulum_radian)) / 2

        energy_penalty = 2.0 * -1.0 * (abs(self.pendulum_velocity) + abs(self.motor_velocity)) / 100

        reward = position_reward + energy_penalty

        reward = max(0.0, reward)

        return reward
