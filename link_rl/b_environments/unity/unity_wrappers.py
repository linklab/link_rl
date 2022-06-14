import copy

import gym
from gym.spaces import Box
from gym_unity.envs import UnityToGymWrapper
from mlagents_envs.base_env import ActionTuple
from torchvision import transforms as T
import numpy as np
import torch

class ProcessFrame(gym.ObservationWrapper):
    def __init__(self, env, sh):
        super(ProcessFrame, self).__init__(env)

        obs_shape = self.observation_space.shape[:2]
        self.observation_space = Box(low=0, high=255, shape=(obs_shape[2], obs_shape[0], obs_shape[1]))

    def observation(self, observation):
        return observation.transpose((2, 0, 1))


class GrayScaleObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        obs_shape = self.observation_space.shape[:2]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def permute_orientation(self, observation):
        # permute [H, W, C] array to [C, H, W] tensor
        observation = np.transpose(observation, (2, 0, 1))
        observation = torch.tensor(observation.copy(), dtype=torch.float)
        return observation

    def observation(self, observation):
        observation = self.permute_orientation(observation)
        transform = T.Grayscale()
        observation = transform(observation)
        return observation


class ResizeObservation(gym.ObservationWrapper):
    def __init__(self, env, shape):
        super(ResizeObservation, self).__init__(env)
        if isinstance(shape, int):
            self.shape = (shape, shape)
        else:
            self.shape = tuple(shape)

        obs_shape = self.shape + self.observation_space.shape[2:]
        self.observation_space = Box(low=0, high=255, shape=(obs_shape))

    def observation(self, observation):
        transforms = T.Compose(
            [T.Resize(self.shape), T.Normalize(0, 255)]
        )
        observation = transforms(observation).squeeze(0)
        return observation


class TransformReward(gym.RewardWrapper):
    def __init__(self, env):
        super().__init__(env)

    def reward(self, reward):
        return np.tanh(reward)
