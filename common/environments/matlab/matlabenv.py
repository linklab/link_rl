import math
from abc import ABC
import matlab.engine
import gym
from gym import spaces
import time
import numpy as np

from matlabcode import SimulinkPlant
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
        self.obs_degree = [None, None]
        self.next_obs_degree = [None, None]

    def pause(self):
        self.plant.conncectpause()

    def start(self):
        self.plant.connectToMatlab()

    def disconnect(self):
        self.plant.disconnect()

    def reset(self):
        self.plant.connectStart()
        self.episode_steps = 0
        self.q, self.q1, self.w, self.w1 = self.plant.getHistory()
        obs = (self.q/math.pi, self.q1, self.w, self.w1)
        self.obs_degree[0] = self.next_obs_degree[0] = self.convert_radian_to_degree(np.round(obs, decimals=4)[0] * math.pi)
        self.obs_degree[1] = self.next_obs_degree[1] = self.convert_radian_to_degree(np.round(obs, decimals=4)[1])
        return obs

    def step(self, action):
        self.plant.simulate(action)
        self.q, self.q1, self.w, self.w1 = self.plant.getHistory()

        self.episode_steps += 1
        self.total_steps += 1
        global a
        # if self.episode_steps >= 100:
        #     done = True
        #     reward = 10
        #     self.plant.connectStop()
        # elif self.q > 3.49065 or self.q <2.79252:
        #     done = True
        #     self.plant.connectStop()
        #     if self.episode_steps < 11:
        #         reward = -10
        #     else:
        #         reward = 1
        # else:
        #     done = False
        #     reward = 1

        done_conditions = [
            self.episode_steps >= 2000,
            self.q < 3.054326, # 175
            self.q > 3.228859   # 185

        ]

        self.obs_degree[0] = self.next_obs_degree[0]
        self.obs_degree[1] = self.next_obs_degree[1]

        next_obs, reward, info = (self.q/math.pi, self.q1, self.w, self.w1), 0.1, None

        self.next_obs_degree[0] = self.convert_radian_to_degree(np.round(next_obs, decimals=4)[0] * math.pi)
        self.next_obs_degree[1] = self.convert_radian_to_degree(np.round(next_obs, decimals=4)[1])

        if any(done_conditions):
            done = True
            self.plant.connectStop()
        else:
            done = False

        # print("!!!!!!!!!!!!!!!!!!!!!!!!", np.asarray(next_obs))
        return next_obs, reward, done, info

    def render(self, mode='human'):
        pass

    @staticmethod
    def convert_radian_to_degree(radian):
        degree = radian * 180 / math.pi
        return degree

