import math
from abc import ABC
import matlab.engine
import gym
from gym import spaces
import time
import numpy as np
from common.environments.matlab.matlabcode import SimulinkPlant
from config.parameters import PARAMETERS as params
from collections import deque
np.set_printoptions(formatter={'float_kind': lambda x: '{0:0.6f}'.format(x)})

a = 0
class MatlabRotaryInvertedPendulumEnv(gym.Env):
    def __init__(self):
        self.episode_steps = 0
        self.total_steps = 0
        self.q = 0
        self.q1 = 0
        self.w = 0
        self.w1 = 0
        self.plant = SimulinkPlant()
        self.state = None
        self.obs_degree = [None, None]
        self.next_obs_degree = [None, None]
        self.simulation_time = 0.0
        self.state_deque = deque(maxlen=30)
        self.num_continuous_positive_torque = 0
        self.num_continuous_negative_torque = 0
        # self.done_torque_threshold = 0.75

    def pause(self):
        self.plant.conncectpause()

    def start(self):
        self.plant.connectToMatlab()

    def disconnect(self):
        self.plant.disconnect()

    def reset(self):
        self.plant.connectStart()
        self.episode_steps = 0
        self.q, self.q1, self.w, self.w1, self.simulation_time = self.plant.getHistory()
        # self.state = (math.cos(self.q), math.sin(self.q), self.w, math.cos(self.q1), math.sin(self.q1), self.w1)
        self.state = (math.cos(self.q), math.sin(self.q), self.w, 0.0)
        self.num_continuous_positive_torque = 0
        self.num_continuous_negative_torque = 0
        # self.obs_degree[0] = self.next_obs_degree[0] = self.convert_radian_to_degree(np.round(self.state, decimals=4)[0] * math.pi)
        # self.obs_degree[1] = self.next_obs_degree[1] = self.convert_radian_to_degree(np.round(self.state, decimals=4)[1])

        # print("q: {0:7.4}, w: {1:7.4f}, time: {2} -- RESET".format(
        #     self.q, self.w, self.simulation_time
        # ))
        return np.array(self.state)

    def step(self, action):
        self.plant.simulate(action)
        self.q, self.q1, self.w, self.w1, self.simulation_time = self.plant.getHistory()
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

        # radian을 0과 2 * math.pi 사이 값(양수)으로 조정
        if abs(self.q) > 2 * math.pi:
            q_ = abs(self.q) % (2 * math.pi)
        else:
            q_ = abs(self.q)

        # radian을 0과 math.pi 사이 값(양수)으로 조정: 3 * math.pi / 2 -->  2 * math.pi - 3 * math.pi / 2 --> math.pi / 2
        if q_ > math.pi:
            adjusted_radian = 2 * math.pi - q_
        else:
            adjusted_radian = q_

        # self.state = (math.cos(self.q), math.sin(self.q), self.w, math.cos(self.q1), math.sin(self.q1), self.w1)
        self.state = (math.cos(self.q), math.sin(self.q), self.w, action[-1])
        self.state_deque.append(self.state)

        info = [None]

        done_conditions = [
            self.episode_steps >= 2000 if params.TEAMVIEWER else self.episode_steps >=1000
            # self.num_continuous_positive_torque >= 30,
            # self.num_continuous_negative_torque >= 30
        ]

        # if any(done_conditions):
        #     done = True
        #     if self.num_continuous_positive_torque >= 15 or self.num_continuous_negative_torque >= 15:
        #         reward = -100000.0
        #     else:
        #         reward = self._ordinary_reward(adjusted_radian, action)


        # if any(done_conditions):
        #     done = True
        #     reward = self._ordinary_reward(
        #         adjusted_radian, action, self.num_continuous_positive_torque, self.num_continuous_negative_torque
        #     )
        #     self.plant.connectStop()
        # else:
        #     done = False
        #     reward = self._ordinary_reward(
        #         adjusted_radian, action, self.num_continuous_positive_torque, self.num_continuous_negative_torque
        #     )
        if any(done_conditions):
            done = True
            if params.CH:
                reward = self.CH_ordinary_reward(
                    adjusted_radian, action, self.num_continuous_positive_torque, self.num_continuous_negative_torque
                )
            else:
                reward = self._ordinary_reward(
                    adjusted_radian, action, self.num_continuous_positive_torque, self.num_continuous_negative_torque
                )
            self.plant.connectStop()

        else:
            done = False
            if params.CH:
                reward = self.CH_ordinary_reward(
                    adjusted_radian, action, self.num_continuous_positive_torque, self.num_continuous_negative_torque
                )
            else:
                reward = self._ordinary_reward(
                    adjusted_radian, action, self.num_continuous_positive_torque, self.num_continuous_negative_torque
                )
        if not isinstance(reward, float):
            reward = reward[-1]

        # print("action: {0}, q: {1:7.4}, w: {2:7.4f}, adjusted_radian: {3:7.4f}, reward: {4:10.4f}, time: {5}".format(
        #     action, self.q, self.w, adjusted_radian, reward, self.simulation_time
        # ))

        return np.array(self.state_deque[-1]), reward, done, info

    def _ordinary_reward(self, adjusted_radian, action, num_continuous_positive_torque, num_continuous_negative_torque):
        # reward = -((math.pi - adjusted_radian) ** 2 + 0.1 * (self.w ** 2) + 0.001 * (action ** 2))
        if adjusted_radian < math.pi / 2:
            reward = 0.0
        else:
            reward = adjusted_radian

        reward -= num_continuous_positive_torque * 0.01
        reward -= num_continuous_negative_torque * 0.01

        return reward

    def CH_ordinary_reward(self, adjusted_radian, action, num_continuous_positive_torque,
                         num_continuous_negative_torque):
        # reward = -((math.pi - adjusted_radian) ** 2 + 0.1 * (self.w ** 2) + 0.001 * (action ** 2))
        if adjusted_radian < math.pi / 2:
            reward = 0.0 - abs(np.tanh(self.w1)) * 0.1
        else:
            reward = adjusted_radian - abs(np.tanh(self.w1)) * 0.1

        reward -= num_continuous_positive_torque * 0.01
        reward -= num_continuous_negative_torque * 0.01
        
        return reward

    def render(self, mode='human'):
        pass

    @staticmethod
    def convert_radian_to_degree(radian):
        degree = radian * 180 / math.pi
        return degree

