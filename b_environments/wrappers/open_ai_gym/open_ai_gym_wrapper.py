import gym
import numpy as np


class ObservationZeroWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

    def observation(self, observation):
        observation = np.zeros_like(observation)
        return observation
