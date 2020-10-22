import math
from abc import ABC
import matlab.engine
import gym
from gym import spaces
import time
import numpy as np
from common.environments.matlab.matlabcode import SimulinkPlant

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
        self.state = (math.cos(self.q), math.sin(self.q), self.w)
        # self.obs_degree[0] = self.next_obs_degree[0] = self.convert_radian_to_degree(np.round(self.state, decimals=4)[0] * math.pi)
        # self.obs_degree[1] = self.next_obs_degree[1] = self.convert_radian_to_degree(np.round(self.state, decimals=4)[1])

        print(self.q, "reset")
        return np.array(self.state)

    def step(self, action):
        self.plant.simulate(action)
        self.q, self.q1, self.w, self.w1 = self.plant.getHistory()
        self.episode_steps += 1
        self.total_steps += 1

        done_conditions = [
            self.episode_steps >= 1000,
            self.w > 300,
            self.w < -300
        ]

        print(self.q)

        self.state = (math.cos(self.q), math.sin(self.q), self.w)
        reward = -((math.pi - self.q) ** 2 + 0.1 * (self.w ** 2) + 0.001 * (action ** 2))
        info = [None]

        if any(done_conditions):
            done = True
            self.plant.connectStop()
        else:
            done = False

        return np.array(self.state), reward[-1], done, info

    def render(self, mode='human'):
        pass

    @staticmethod
    def convert_radian_to_degree(radian):
        degree = radian * 180 / math.pi
        return degree

