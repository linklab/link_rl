import math
from abc import ABC
from enum import Enum

import matlab.engine
import gym
import numpy as np
from common.environments.matlab.matlabcode import SimulinkPlant
from config.parameters import PARAMETERS as params
np.set_printoptions(formatter={'float_kind': lambda x: '{0:0.6f}'.format(x)})


class MatlabRotaryInvertedPendulumEnv(gym.Env):
    def __init__(self, action_min, action_max, env_reset=True):
        self.episode_steps = 0
        self.total_steps = 0
        self.env_reset = env_reset

        self.pendulum_position = 0
        self.pendulum_velocity = 0
        self.motor_position = 0
        self.motor_velocity = 0

        self.plant = SimulinkPlant()
        self.obs_degree = [None, None]
        self.next_obs_degree = [None, None]
        self.simulation_time = 0.0
        self.num_continuous_positive_torque = 0
        self.num_continuous_negative_torque = 0

        self.too_much_rotate = False
        # self.done_torque_threshold = 0.75

        self.max_velocity = 100.0

        self.action_space = gym.spaces.Box(
            low=action_min, high=action_max, shape=(1,),
            dtype=np.float32
        )

        #high = np.array([1., 1., self.max_velocity, 1., 1., action_max, 1.0], dtype=np.float32)
        high = np.array([1., 1., self.max_velocity, 1., 1.], dtype=np.float32)
        low = high * -1.0
        self.observation_space = gym.spaces.Box(
            low=low, high=high,
            dtype=np.float32
        )

        self.n_states = self.observation_space.shape[0]
        self.n_actions = self.action_space.shape[0]

        self.current_status = None

        self.count_continuous_uprights = 0
        self.is_upright = False
        self.initial_motor_position = 0.0

        self.episode_position_reward_list = []
        self.episode_pendulum_velocity_reward_list = []
        self.episode_action_reward_list = []

    def get_n_states(self):
        n_states = self.observation_space.shape[0]
        return n_states

    def get_n_actions(self):
        n_actions = self.action_space.shape[0]
        return n_actions

    @property
    def action_meanings(self):
        action_meanings = ["Joint effort",]
        return action_meanings

    def pause(self):
        self.plant.conncectpause()

    def start(self):
        self.plant.connectToMatlab()

    def disconnect(self):
        self.plant.disconnect()

    def reset(self):
        self.episode_steps = 0

        self.episode_position_reward_list.clear()
        self.episode_pendulum_velocity_reward_list.clear()
        self.episode_action_reward_list.clear()

        if self.env_reset:
            self.plant.connectStart()

        self.pendulum_position, self.motor_position, self.pendulum_velocity, self.motor_velocity, self.simulation_time = self.plant.getHistory()

        self.update_current_state(adjusted_radian=0.0)

        state = (
            math.cos(self.pendulum_position),
            math.sin(self.pendulum_position),
            self.pendulum_velocity,
            math.cos(0.0),  # 1.0
            math.sin(0.0),  # 0.0
            #self.motor_velocity,
        )

        self.num_continuous_positive_torque = 0
        self.num_continuous_negative_torque = 0

        # print("q: {0:7.4}, w: {1:7.4f}, time: {2} -- RESET".format(
        #     self.pendulum_position, self.pendulum_velocity, self.simulation_time
        # ))

        self.too_much_rotate = False

        self.count_continuous_uprights = 0
        self.is_upright = False
        self.initial_motor_position = self.motor_position

        return state

    def pendulum_position_to_adjusted_radian(self):
        # radian을 0과 2 * math.pi 사이 값(양수)으로 조정
        if abs(self.pendulum_position) > 2 * math.pi:
            q_ = abs(self.pendulum_position) % (2 * math.pi)
        else:
            q_ = abs(self.pendulum_position)

        # radian을 0과 math.pi 사이 값(양수)으로 조정: 3 * math.pi / 2 -->  2 * math.pi - 3 * math.pi / 2 --> math.pi / 2
        if q_ > math.pi:
            adjusted_radian = 2 * math.pi - q_
        else:
            adjusted_radian = q_

        return adjusted_radian

    def update_current_state(self, adjusted_radian):
        if params.CH:
            if math.pi - math.radians(3) < adjusted_radian <= math.pi:
                self.count_continuous_uprights += 1
            else:
                self.count_continuous_uprights = 0
        else:
            if math.pi - math.radians(12) < adjusted_radian <= math.pi:
                self.count_continuous_uprights += 1
            else:
                self.count_continuous_uprights = 0

        if self.count_continuous_uprights >= 1:
            self.is_upright = True
        else:
            self.is_upright = False

    def step(self, action):
        if type(action) is np.ndarray:
            action = action[0]

        self.plant.simulate(action)
        self.pendulum_position, self.motor_position, self.pendulum_velocity, self.motor_velocity, self.simulation_time = self.plant.getHistory()
        self.episode_steps += 1
        self.total_steps += 1

        if params.CH:
            pass
        else:
            if action > 0:
                self.num_continuous_positive_torque += 1
            else:
                self.num_continuous_positive_torque = 0

            if action < 0:
                self.num_continuous_negative_torque += 1
            else:
                self.num_continuous_negative_torque = 0

        #print(self.motor_position, math.cos(self.motor_position), math.sin(self.motor_position))

        if abs(self.initial_motor_position - self.motor_position) > math.pi * 2:
            self.too_much_rotate = True

        done_conditions = [
            self.episode_steps >= 500 and not self.is_upright,
            self.too_much_rotate and not self.is_upright
            # self.num_continuous_positive_torque >= 30,
            # self.num_continuous_negative_torque >= 30
        ]

        adjusted_radian = self.pendulum_position_to_adjusted_radian()

        # print("action: {0}, q: {1:7.4}, w: {2:7.4f}, adjusted_radian: {3:7.4f}, reward: {4:10.4f}, time: {5}".format(
        #     action, self.pendulum_position, self.pendulum_velocity, adjusted_radian, reward, self.simulation_time
        # ))

        self.update_current_state(adjusted_radian)

        if any(done_conditions):
            done = True
            if params.CH:
                reward = self.CH_ordinary_reward(
                    adjusted_radian, action, self.num_continuous_positive_torque, self.num_continuous_negative_torque
                )
            else:
                reward = self.get_reward(adjusted_radian)

            info = {
                "episode_position_reward_list": sum(self.episode_position_reward_list),
                "episode_pendulum_velocity_reward": sum(self.episode_pendulum_velocity_reward_list),
                "episode_action_reward": sum(self.episode_action_reward_list)
            }

            if self.env_reset:
                self.plant.connectStop()
        else:
            done = False
            if params.CH:
                reward = self.CH_ordinary_reward(
                    adjusted_radian, action, self.num_continuous_positive_torque, self.num_continuous_negative_torque
                )
            else:
                reward = self.get_reward(adjusted_radian)

            info = {}

        state = (
            math.cos(self.pendulum_position),
            math.sin(self.pendulum_position),
            self.pendulum_velocity,
            math.cos(self.initial_motor_position - self.motor_position),
            math.sin(self.initial_motor_position - self.motor_position),
            #self.motor_velocity,
        )

        return state, reward, done, info

    def get_reward(self, adjusted_radian):
        # if self.too_much_rotate:
        #     position_reward = -1.0
        #     energy_penalty = 0.0
        # else:

        if self.is_upright:
            position_reward = adjusted_radian / math.pi  # math.pi - math.radians(12) ~ math.pi
        elif adjusted_radian > math.pi / 2.0:
            position_reward = adjusted_radian / (math.pi * 2.0)
        else:
            position_reward = 0.0

        energy_penalty = -1.0 * abs(self.pendulum_velocity) / 100

        self.episode_position_reward_list.append(position_reward)
        self.episode_pendulum_velocity_reward_list.append(energy_penalty)
        self.episode_action_reward_list.append(0.0)

        reward = position_reward + energy_penalty

        reward = max(0.0, reward)

        print(position_reward, energy_penalty, reward)

        return reward

    # def get_reward(self, adjusted_radian, action):
    #     #### 1) position_reward
    #     if self.too_much_rotate:
    #         position_reward = -100.0
    #     else:
    #         if adjusted_radian < math.pi / 2:
    #             position_reward = 0.0
    #         else:
    #             position_reward = adjusted_radian
    #
    #     self.episode_position_reward_list.append(position_reward)
    #
    #     #### 2) pendulum_velocity 보상 & 3) action 보상
    #     if self.current_status in [Status.BALANCING, Status.SWING_UP_TO_BALANCING]:
    #         pendulum_velocity_reward = -0.001 * self.pendulum_velocity ** 2
    #         self.episode_pendulum_velocity_reward_list.append(pendulum_velocity_reward)
    #
    #         action_reward = -50.0 * abs(action)
    #         self.episode_action_reward_list.append(action_reward)
    #
    #         reward = position_reward + pendulum_velocity_reward + action_reward
    #     else:
    #         reward = position_reward
    #
    #     reward /= 1000.0
    #
    #     return reward

    def CH_ordinary_reward(self, adjusted_radian, action, num_continuous_positive_torque,
                         num_continuous_negative_torque):
        # reward = -((math.pi - adjusted_radian) ** 2 + 0.1 * (self.pendulum_velocity ** 2) + 0.001 * (action ** 2))
        if adjusted_radian < math.pi / 2:
            reward = 0.0 - abs(np.tanh(self.motor_velocity)) * 0.1
        else:
            reward = adjusted_radian - abs(np.tanh(self.motor_velocity)) * 0.1

        reward -= num_continuous_positive_torque * 0.01
        reward -= num_continuous_negative_torque * 0.01
        
        return reward

    def render(self, mode='human'):
        pass

    @staticmethod
    def convert_radian_to_degree(radian):
        degree = radian * 180 / math.pi
        return degree

