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

RIP_SERVER = '10.0.0.4'
GOAL_ANGLE = params.GOAL_ANGLE

def get_quanser_rip_observation_space():
    low = np.array([0, 0, 0, 0, 0, 0], dtype=np.float32)
    high = np.array([1., 1., 500., 1., 1., 500,], dtype=np.float32)

    observation_space = gym.spaces.Box(
        low=low, high=high, dtype=np.float32
    )
    n_states = observation_space.shape[0]
    return observation_space, n_states


def get_quanser_rip_action_info(params):
    if params.RL_ALGORITHM in [RLAlgorithmName.DQN_V0]:
        action_index_to_voltage = list(np.array([
            -900, -500, 0, 500, 900
        ]))
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


class AbjustAngleEnv(gym.Env):
    def __init__(self, mode=AgentMode.TRAIN):
        super(AbjustAngleEnv, self).__init__()
        self.params = params

        self.previous_time = 0

        self.episode = 0

        self.env_reset = False

        self.reward = 0

        self.step_idx = 0
        self.state = []
        self.episode_steps = 0

        self.count_continuous_uprights = 0
        self.is_upright = False

        self.motor_radian = 0
        self.motor_velocity = 0
        self.pendulum_radian = 0
        self.pendulum_velocity = 0

        self.is_motor_limit = False

        self.unit_time = self.params.UNIT_TIME
        self.over_unit_time = 0

        if mode == AgentMode.PLAY:
            self.max_episode_step = self.params.MAX_EPISODE_STEP_AT_PLAY
        else:
            self.max_episode_step = self.params.MAX_EPISODE_STEP

        self.initial_motor_radian = 0.0
        #==================observation==========================================================
        self.observation_space, self.n_states = get_quanser_rip_observation_space()
        #=======================================================================================

        #==================action===============================================================
        self.action_space, self.n_actions, self.action_index_to_voltage = get_quanser_rip_action_info(self.params)
        #=======================================================================================

        channel = grpc.insecure_channel('{0}:50051'.format(RIP_SERVER))
        self.server_obj = quanser_service_pb2_grpc.QuanserRIPStub(channel)

    def reset(self):
        self.episode_steps = 0
        self.reward = 0
        self.is_motor_limit = False

        quanser_response = self.server_obj.reset_sync(QuanserRequest(value=0.0))

        if quanser_response.message != "RESET_SYNC":
            raise ValueError()

        self.motor_radian = quanser_response.motor_radian
        self.motor_velocity = quanser_response.motor_velocity
        self.pendulum_radian = quanser_response.pendulum_radian
        self.pendulum_velocity = quanser_response.pendulum_velocity

        if self.episode % 5 == 0:
            print("*RESET PENDULUM RADIAN : {0:1.3f}".format(self.pendulum_radian))

        self.state = [
            math.cos(self.pendulum_radian),
            math.sin(self.pendulum_radian),
            self.pendulum_velocity / params.VELOCITY_STATE_DENOMINATOR,
            # math.cos(self.initial_motor_radian - self.motor_radian),
            # math.sin(self.initial_motor_radian - self.motor_radian),
            math.cos(quanser_response.motor_radian),
            math.sin(quanser_response.motor_radian),
            self.motor_velocity / params.VELOCITY_STATE_DENOMINATOR
        ]

        # wait_time = 1 if self.episode == 0 else 15  # if self.episode % 10 == 0 else 3

        previousTime = time.perf_counter()
        time_done = False
        while not time_done:
            currentTime = time.perf_counter()
            if currentTime - previousTime >= self.unit_time:
                time_done = True
            time.sleep(0.0001)

        self.episode += 1

        return self.state

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

        # if self.step_idx % 50 == 0:
        #     self.action_ = -self.action_

        quanser_response = self.server_obj.step_sync(QuanserRequest(value=action))

        self.motor_radian = quanser_response.motor_radian
        self.motor_velocity = quanser_response.motor_velocity
        self.pendulum_radian = quanser_response.pendulum_radian
        self.pendulum_velocity = quanser_response.pendulum_velocity
        self.step_syncronize = quanser_response.is_motor_limit

        self.state = [
            math.cos(self.pendulum_radian),
            math.sin(self.pendulum_radian),
            self.pendulum_velocity / params.VELOCITY_STATE_DENOMINATOR,
            # math.cos(self.initial_motor_radian - self.motor_radian),
            # math.sin(self.initial_motor_radian - self.motor_radian),
            math.cos(quanser_response.motor_radian),
            math.sin(quanser_response.motor_radian),
            self.motor_velocity / params.VELOCITY_STATE_DENOMINATOR
        ]

        next_state = self.state

        #=======================reward================================================================
        self.reward = self.get_reward()
        #=============================================================================================
        self.step_idx += 1
        self.episode_steps += 1

        done, info = self.__isDone()

        # print("pendulum radian : {0}, motor radian: {1}, reward: {2}, pendulum_velocity : {3} \n\n".format(
        #     self.pendulum_radian, self.motor_radian, self.is_motor_limit, self.pendulum_velocity
        # ))

        return next_state, self.reward, done, info

    def __isDone(self):
        info = {}

        def insert_to_info(s):
            info["result"] = s

        if self.episode_steps >= self.max_episode_step: # 5000 * 25ms (0.025sec.) = 125 sec.
            insert_to_info("*** Success ***")
            return True, info
        # elif self.is_motor_limit:
        #     print(self.motor_radian, " !!!!!!!")
        #     insert_to_info("*** Limit position ***")
        #     return True, info
        elif self.is_motor_limit:#abs(self.motor_radian) > math.radians(90) or self.is_motor_limit:
            insert_to_info("***motor_radian exceed 90***")
            return True, info
        elif not self.step_syncronize:
            insert_to_info("**not_sync_step**")
            return True, info
        else:
            insert_to_info("")
            return False, info

    def get_reward(self):
        position_reward = math.pi - abs(self.pendulum_radian)  # math.pi - math.radians(12) ~ math.pi
        energy_penalty = -1.0 * (abs(self.pendulum_velocity) + abs(self.motor_velocity)) / 100

        inverted_reward = (position_reward + energy_penalty)/params.REWARD_DENOMINATOR
        angle_reward = 1 - abs(GOAL_ANGLE - math.degrees(self.motor_radian))/(abs(GOAL_ANGLE)+90.0)

        reward = (inverted_reward + angle_reward)/2.0
        reward = max(0.000001, reward)

        print(inverted_reward, angle_reward, reward)

        return reward

    def render(self):
        pass