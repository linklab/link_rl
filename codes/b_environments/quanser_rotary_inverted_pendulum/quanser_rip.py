import time
import math

import gym
from gym import spaces
import numpy as np
import grpc

# MQTT Topic for RIP
from codes.b_environments.quanser_rotary_inverted_pendulum import quanser_service_pb2_grpc
from codes.e_utils.names import RLAlgorithmName
from common.environments.environment import Environment
from codes.a_config.parameters import PARAMETERS as params
from codes.b_environments.quanser_rotary_inverted_pendulum.quanser_service_pb2 import QuanserRequest

STATE_SIZE = 6

balance_motor_power_list = [-60., 0., 60.]

RIP_SERVER = '10.0.0.5'


def get_quanser_rip_observation_space():
    low = np.array([0, 0, 0, 0, 0, 0], dtype=np.float32)
    high = np.array([1., 1., 500., 1., 1., 500,], dtype=np.float32)

    observation_space = gym.spaces.Box(
        low=low, high=high, dtype=np.float32
    )
    n_states = observation_space.shape[0]
    return observation_space, n_states


def get_quanser_rip_action_space(params, action_index_to_voltage=None):
    if params.RL_ALGORITHM in [RLAlgorithmName.DQN_V0]:
        action_space = gym.spaces.Discrete(len(action_index_to_voltage))
        n_actions = action_space.n
    else:
        action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(1,),
            dtype=np.float32
        )
        n_actions = action_space.shape[0]

    return action_space, n_actions


class EnvironmentQuanserRIP(gym.Env):
    def __init__(self):
        super(EnvironmentQuanserRIP, self).__init__()
        self.episode = 0

        self.params = params

        self.previous_time = time.perf_counter()
        self.state_space_shape = (STATE_SIZE,)
        self.action_space_shape = (len(balance_motor_power_list),)

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

        self.initial_motor_radian = 0.0
        #==================observation==========================================================
        self.observation_space, self.n_states = get_quanser_rip_observation_space()
        #=======================================================================================

        #==================action===============================================================
        self.action_space, self.n_actions = get_quanser_rip_action_space(self.params, None)
        #=======================================================================================

        channel = grpc.insecure_channel('{0}:50051'.format(RIP_SERVER))
        self.server_obj = quanser_service_pb2_grpc.QuanserRIPStub(channel)

    def get_n_states(self):
        n_states = 6
        return n_states
    #
    # def get_n_actions(self):
    #     n_actions = 3
    #     return n_actions

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

    # def pendulum_reset(self):
    #     quanser_response = self.server_obj.step(QuanserRequest(value=0.0))
    #     if quanser_response.message != "PENDULUM_RESET":
    #         raise ValueError()

    def reset(self):
        self.episode_steps = 0
        self.reward = 0
        self.is_motor_limit = False

        quanser_response = self.server_obj.reset(QuanserRequest(value=0.0))
        if quanser_response.message != "RESET":
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
            self.pendulum_velocity,
            # math.cos(0.0),
            # math.sin(0.0),
            math.cos(quanser_response.motor_radian),
            math.sin(quanser_response.motor_radian),
            self.motor_velocity
        ]
        # wait_time = 1 if self.episode == 0 else 15  # if self.episode % 10 == 0 else 3
        wait_time = 1
        previousTime = time.perf_counter()
        time_done = False
        while not time_done:
            currentTime = time.perf_counter()
            if currentTime - previousTime >= wait_time:
                time_done = True
            time.sleep(0.001)

        self.initial_motor_radian = self.motor_radian
        self.is_motor_limit = False
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

        print(self.step_idx, action, step_time)
        self.previous_time = time.perf_counter()

        if self.step_idx % 100000 == 0:
            print("*OVER UNIT TIME STEP NUMBER :", self.over_unit_time)
        #######################################################

        if type(action) is np.ndarray:
            action = action[0]

        #motor_power = float(action)
        # if self.step_idx % 100 == 0:
        #     print(action)

        #==================== Grpc and use sample time========================================
        # previous_time = time.perf_counter()
        # print(action)
        quanser_response = self.server_obj.step(QuanserRequest(value=action))
        # print(motor_power)
        # if quanser_response.message != "STEP":
        #     raise ValueError()
        # while True:
        #     current_time = time.perf_counter()
        #     if current_time - previous_time >= 25 / 1000:
        #         break
        #     time.sleep(0.0001)
        #=====================================================================================

        self.motor_radian = quanser_response.motor_radian
        self.motor_velocity = quanser_response.motor_velocity
        self.pendulum_radian = quanser_response.pendulum_radian
        self.pendulum_velocity = quanser_response.pendulum_velocity
        self.is_motor_limit = quanser_response.is_motor_limit
        # print("motor: ", self.motor_radian)
        # print("===========transfer time : {0:1.6f} action : {1:3.2f}".format(previous_time - time.perf_counter(), action))
        # print("motor radian : {0:1.3f}, motor velocity : {1:1.3f}, pendulum radian : {2:1.3f}, pendulum velocity : {3:1.3f}".format(
        #     self.motor_radian, self.motor_velocity,
        #       self.pendulum_radian, self.pendulum_velocity
        #     ))
        # print("is motor limit", self.is_motor_limit)

        self.state = [
            math.cos(self.pendulum_radian),
            math.sin(self.pendulum_radian),
            self.pendulum_velocity,
            # math.cos(self.initial_motor_radian - self.motor_radian),
            # math.sin(self.initial_motor_radian - self.motor_radian),
            math.cos(quanser_response.motor_radian),
            math.sin(quanser_response.motor_radian),
            self.motor_velocity
        ]
        next_state = np.asarray(self.state)

        self.update_current_state()

        done, info = self.__isDone()

        #=======================reward================================================================
        self.reward = self.get_reward(action)
        # print(self.reward, self.pendulum_radian)
        #=============================================================================================
        self.step_idx += 1
        self.episode_steps += 1

        # print("pendulum radian : {0}, motor radian: {1}, reward: {2}, pendulum_velocity : {3} \n\n".format(
        #     self.pendulum_radian, self.motor_radian, self.is_motor_limit, self.pendulum_velocity
        # ))

        return next_state, self.reward, done, info

    def __isDone(self):
        info = {}

        def insert_to_info(s):
            info["result"] = s

        if self.episode_steps >= self.params.MAX_EPISODE_STEP: # 5000 * 25ms (0.025sec.) = 125 sec.
            insert_to_info("*** Success ***")
            return True, info
        # elif self.is_motor_limit:
        #     print(self.motor_radian, " !!!!!!!")
        #     insert_to_info("*** Limit position ***")
        #     return True, info
        elif self.is_motor_limit:#abs(self.motor_radian) > math.radians(90) or self.is_motor_limit:
            insert_to_info("***motor_radian exceed 90***")
            return True, info
        else:
            insert_to_info("")
            return False, info

    def update_current_state(self):
        if abs(self.pendulum_radian) < math.radians(90):
            self.count_continuous_uprights += 1
        else:
            self.count_continuous_uprights = 0

        if self.count_continuous_uprights >= 1:
            self.is_upright = True
        else:
            self.is_upright = False

    # def get_reward(self):
    #     if abs(self.pendulum_radian) > math.radians(30):
    #         position_reward = 0.0
    #     else:
    #         if self.is_upright:
    #             position_reward = math.pi - abs(self.pendulum_radian)  # math.pi - math.radians(12) ~ math.pi
    #         else:
    #             if self.is_motor_limit:
    #                 position_reward = 0.0
    #             else:
    #                 position_reward = (math.pi - abs(self.pendulum_radian)) / 2
    #
    #     energy_penalty = 2.0 * -1.0 * (abs(self.pendulum_velocity) + abs(self.motor_velocity)) / 100
    #
    #     reward = position_reward + energy_penalty
    #
    #     reward = max(0.0, reward)
    #
    #     return reward

    def get_reward(self, action):
        # print(self.is_upright, self.pendulum_radian, math.radians(90), action)
        if self.is_upright:
            position_reward = math.pi - abs(self.pendulum_radian)# math.pi - math.radians(12) ~ math.pi
        else:
            if self.is_motor_limit:
                position_reward = 0.0
            else:
                position_reward = (math.pi - abs(self.pendulum_radian)) / 2
                # position_reward = 1 + math.cos(self.pendulum_radian)


        energy_penalty = 1.0 * -1.0 * (abs(self.pendulum_velocity) + abs(self.motor_velocity)) / 2000

        reward = position_reward + energy_penalty

        reward = max(0.0, reward)

        # open ai pendulum reward = -(theta^2 + 0.1theta_dt^2 + 0.001action^2)
        # reward = -((self.pendulum_radian**2) + 0.1*((self.pendulum_velocity/100)**2) + 0.001*(((2*action)/params.ACTION_SCALE)**2))

        #print(position_reward, energy_penalty, reward)

        return reward

    def render(self):
        pass