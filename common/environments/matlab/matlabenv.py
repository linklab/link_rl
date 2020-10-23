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
        self.simulation_time = 0.0
        self.dt = 0.005
        self.degree = 0.0

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
        self.state = (math.cos(self.q), math.sin(self.q), self.w)
        # self.obs_degree[0] = self.next_obs_degree[0] = self.convert_radian_to_degree(np.round(self.state, decimals=4)[0] * math.pi)
        # self.obs_degree[1] = self.next_obs_degree[1] = self.convert_radian_to_degree(np.round(self.state, decimals=4)[1])

        self.degree = 0.0
        print("q: {0:7.4}, w: {1:7.4f}, degree: {2:7.4f}, time: {3} -- RESET".format(
            self.q, self.w, self.degree, self.simulation_time
        ))
        return np.array(self.state)

    def step(self, action):
        self.plant.simulate(action)
        self.q, self.q1, self.w, self.w1, self.simulation_time = self.plant.getHistory()
        self.episode_steps += 1
        self.total_steps += 1

        done_conditions = [
            self.episode_steps >= 1000,
            self.w > 300,
            self.w < -300
        ]

        self.degree = self.degree + self.w * self.dt

        # radian을 0과 math.pi 사이 값으로 조정
        if abs(self.q) > math.pi:
            adjusted_radian = 2 * math.pi - abs(self.q)
        else:
            adjusted_radian = self.q

        self.state = (math.cos(self.q), math.sin(self.q), self.w)

        reward = -((math.pi - adjusted_radian) ** 2 + 0.1 * (self.w ** 2) + 0.001 * (action ** 2))

        if not isinstance(reward, float):
            reward = reward[-1]

        print("q: {0:7.4}, w: {1:7.4f}, degree: {2:7.4f}, radian: {3:7.4f}, reward: {4:10.4f}, time: {4}".format(
            self.q, self.w, self.degree, adjusted_radian, reward, self.simulation_time
        ))

        info = [None]

        if any(done_conditions):
            done = True
            self.plant.connectStop()
        else:
            done = False

        return np.array(self.state), reward, done, info

    def render(self, mode='human'):
        pass

    @staticmethod
    def convert_radian_to_degree(radian):
        degree = radian * 180 / math.pi
        return degree

