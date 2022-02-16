import gym
from gym import spaces


class ProcessFrame(gym.ObservationWrapper):
    def __init__(self, env):
        super(ProcessFrame, self).__init__(env)
        self.env = env
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.observation_space.shape[2], self.observation_space.shape[0], self.observation_space.shape[1]))

    def observation(self, observation):
        return observation.transpose((2, 0, 1))
