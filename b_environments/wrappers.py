import numpy as np
import gym


class CustomObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        assert isinstance(env.observation_space, gym.spaces.Discrete), \
            "Should only be used to wrap Discrete envs."
        self.discrete_observation_space_n = self.observation_space.n
        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=(self.discrete_observation_space_n,)
        )  # [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]

    def observation(self, observation):  # Observation --> One-hot vector
        if observation is None:
            return None
        new_obs = np.zeros(self.discrete_observation_space_n) # [0, 0, 0, 0]
        new_obs[observation] = 1  # [0, 1, 0, 0]
        return new_obs


class CustomRewardWrapper(gym.RewardWrapper):
    def __init__(self, env):
        super().__init__(env)

    def reward(self, reward):
        # modify rew
        if reward == 0.0:
            reward = -1.0
        return reward


class CustomActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)

    def action(self, action):
        if action < 0:
            action = 0
        return action
