import time
import math

import gym
from gym import spaces
import numpy as np
import grpc

# MQTT Topic for RIP
from codes.b_environments.quanser_rotary_inverted_pendulum import quanser_service_pb2_grpc
from codes.e_utils.names import RLAlgorithmName, AgentMode
from common.environments.environment import Environment
from codes.a_config.parameters import PARAMETERS as params
from codes.b_environments.quanser_rotary_inverted_pendulum.quanser_service_pb2 import QuanserRequest

STATE_SIZE = 6

balance_motor_power_list = [-60., 0., 60.]

RIP_SERVER_1 = '10.0.0.5'
RIP_SERVER_2 = '10.0.0.4'

def get_quanser_rip_observation_space():
    low = np.array([0, 0, 0, 0, 0, 0], dtype=np.float32)
    high = np.array([1., 1., 500., 1., 1., 500,], dtype=np.float32)

    observation_space = gym.spaces.Box(
        low=low, high=high, dtype=np.float32
    )
    n_states = observation_space.shape[0]
    return observation_space, n_states


def get_quanser_rip_action_space(params):
    if params.RL_ALGORITHM in [RLAlgorithmName.DQN_V0]:
        action_index_to_voltage = list(np.array([
            -1.0, -0.75, -0.5, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0
        ]) * params.ACTION_SCALE)
        action_space = gym.spaces.Discrete(len(action_index_to_voltage))
        n_actions = action_space.n
    else:
        action_index_to_voltage = None
        action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(1,),
            dtype=np.float32
        )
        n_actions = action_space.shape[0]

    return action_space, n_actions, action_index_to_voltage


class SyncronizeEnv(gym.Env):
    def __init__(self, mode=AgentMode.TRAIN):
        super(SyncronizeEnv, self).__init__()
        self.params = params

        self.previous_time = 0

        self.episode = 0

        self.state_space_shape = (STATE_SIZE,)
        self.action_space_shape = (len(balance_motor_power_list),)

        self.env_reset = False

        self.reward = 0

        self.step_idx = 0
        self.state = []
        self.episode_steps = 0

        self.count_continuous_uprights = 0
        self.is_upright = False

        self.motor_radian_1 = 0
        self.motor_velocity_1 = 0
        self.pendulum_radian_1 = 0
        self.pendulum_velocity_1 = 0

        self.motor_radian_2 = 0
        self.motor_velocity_2 = 0
        self.pendulum_radian_2 = 0
        self.pendulum_velocity_2 = 0

        self.is_motor_limit = False

        self.unit_time = self.params.UNIT_TIME
        self.over_unit_time = 0

        if mode == AgentMode.PLAY:
            self.max_episode_step = 100000000
        else:
            self.max_episode_step = self.params.MAX_EPISODE_STEP

        self.initial_motor_radian = 0.0
        #==================observation==========================================================
        self.observation_space, self.n_states = get_quanser_rip_observation_space()
        #=======================================================================================

        #==================action===============================================================
        self.action_space, self.n_actions, self.action_index_to_voltage = get_quanser_rip_action_space(self.params)
        #=======================================================================================

        channel_1 = grpc.insecure_channel('{0}:50051'.format(RIP_SERVER_1))
        channel_2 = grpc.insecure_channel('{0}:50051'.format(RIP_SERVER_2))
        self.server_obj_1 = quanser_service_pb2_grpc.QuanserRIPStub(channel_1)
        self.server_obj_2 = quanser_service_pb2_grpc.QuanserRIPStub(channel_2)

    def reset(self):
        self.episode_steps = 0
        self.reward = 0
        self.is_motor_limit = False

        quanser_response_1 = self.server_obj_1.reset_sync(QuanserRequest(value=0.0))
        quanser_response_2 = self.server_obj_2.reset_sync(QuanserRequest(value=0.0))

        if quanser_response_1.message != "RESET_SYNC" or quanser_response_2.message != "RESET_SYNC":
            raise ValueError()

        self.motor_radian_1 = quanser_response_1.motor_radian
        self.motor_velocity_1 = quanser_response_1.motor_velocity
        self.pendulum_radian_1 = quanser_response_1.pendulum_radian
        self.pendulum_velocity_1 = quanser_response_1.pendulum_velocity

        self.motor_radian_2 = quanser_response_2.motor_radian
        self.motor_velocity_2 = quanser_response_2.motor_velocity
        self.pendulum_radian_2 = quanser_response_2.pendulum_radian
        self.pendulum_velocity_2 = quanser_response_2.pendulum_velocity

        if self.episode % 5 == 0:
            print("*RESET PENDULUM RADIAN : {0:1.3f}, {1:1.3f}".format(self.pendulum_radian_1, self.pendulum_radian_2))

        print("Quanser_1's info {0:<5.3f}, {1:<5.3f}, {2:<5.3f}, {3:<5.3f}".format(
            self.motor_radian_1, self.motor_velocity_1, self.pendulum_radian_1, self.pendulum_velocity_1
        ))
        print("Quanser_2's info {0:<5.3f}, {1:<5.3f}, {2:<5.3f}, {3:<5.3f}".format(
            self.motor_radian_2, self.motor_velocity_2, self.pendulum_radian_2, self.pendulum_velocity_2
        ))

        self.state = [0, 0, 0, 0, 0, 0]
        # wait_time = 1 if self.episode == 0 else 15  # if self.episode % 10 == 0 else 3

        previousTime = time.perf_counter()
        time_done = False
        while not time_done:
            currentTime = time.perf_counter()
            if currentTime - previousTime >= self.unit_time:
                time_done = True
            time.sleep(0.0001)

        self.episode += 1

        return np.asarray(self.state)

    def step(self, action):
        # current_time = time.perf_counter()
        # print("current_time - self.previous_time", current_time - self.previous_time)
        while True:
            current_time = time.perf_counter()
            if current_time - self.previous_time >= self.unit_time:
                break
            time.sleep(0.0001)

        current_time = time.perf_counter()
        step_time = current_time - self.previous_time

        if step_time > self.unit_time:
            self.over_unit_time += 1

        # print(self.step_idx, action, step_time)
        self.previous_time = time.perf_counter()

        if self.step_idx % 100000 == 0:
            print("*OVER UNIT TIME STEP NUMBER :", self.over_unit_time)
        #######################################################

        if type(action) is np.ndarray:
            action = action[0]
            action = action * params.ACTION_SCALE

        if self.params.RL_ALGORITHM in [RLAlgorithmName.DQN_V0]:
            action = self.action_index_to_voltage[int(action)]

        #motor_power = float(action)
        # if self.step_idx % 100 == 0:
        #     print(action)

        #==================== Grpc and use sample time========================================
        # previous_time = time.perf_counter()
        # print(action)

        quanser_response_1 = self.server_obj_1.step_sync(QuanserRequest(value=0.0))
        quanser_response_2 = self.server_obj_2.step_sync(QuanserRequest(value=0.0))

        self.motor_radian_1 = quanser_response_1.motor_radian
        self.motor_velocity_1 = quanser_response_1.motor_velocity
        self.pendulum_radian_1 = quanser_response_1.pendulum_radian
        self.pendulum_velocity_1 = quanser_response_1.pendulum_velocity

        self.motor_radian_2 = quanser_response_2.motor_radian
        self.motor_velocity_2 = quanser_response_2.motor_velocity
        self.pendulum_radian_2 = quanser_response_2.pendulum_radian
        self.pendulum_velocity_2 = quanser_response_2.pendulum_velocity

        self.state = [0, 0, 0, 0, 0, 0]

        next_state = np.asarray(self.state)

        done, info = self.__isDone()

        #=======================reward================================================================
        self.reward = self.get_reward(action)
        #=============================================================================================
        self.step_idx += 1
        self.episode_steps += 1

        # print("pendulum radian : {0}, motor radian: {1}, reward: {2}, pendulum_velocity : {3} \n\n".format(
        #     self.pendulum_radian, self.motor_radian, self.is_motor_limit, self.pendulum_velocity
        # ))

        print("Quanser_1's info {0:<5.3f}, {1:<5.3f}, {2:<5.3f}, {3:<5.3f}".format(
            self.motor_radian_1, self.motor_velocity_1, self.pendulum_radian_1, self.pendulum_velocity_1
        ))
        print("Quanser_2's info {0:<5.3f}, {1:<5.3f}, {2:<5.3f}, {3:<5.3f}".format(
            self.motor_radian_2, self.motor_velocity_2, self.pendulum_radian_2, self.pendulum_velocity_2
        ))

        return next_state, self.reward, done, info

    def __isDone(self):
        return False, None

    def get_reward(self, action):
        reward = 0
        return reward

    def render(self):
        pass