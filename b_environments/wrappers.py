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

    def reverse_action(self, action):
        pass


class MakeBoxFrozenLake(gym.Wrapper):
    def __init__(self):
        self._generate_random_map()
        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=(4, self.nrow, self.ncol)
        )

    def _generate_random_map(self):
        from gym.envs.toy_text.frozen_lake import generate_random_map
        random_map = generate_random_map(size=6, p=0.25)  # F:H = 0.25:0.75
        env = gym.make("FrozenLake-v1", desc=random_map)
        super().__init__(env)

    def reset(self, **kwargs):
        self._generate_random_map()
        _ = self.env.reset(**kwargs)
        return self.observation()

    def step(self, action):
        _, reward, done, info = self.env.step(action)
        observation = self.observation()
        return observation, reward, done, info

    def observation(self):
        box_obs = np.zeros(shape=(4, self.nrow, self.ncol))

        box_obs[0] = (self.desc == b'S')  # 1-channel: start
        box_obs[1] = (self.desc == b'H')  # 2-channel: hole
        box_obs[2] = (self.desc == b'G')  # 3-channel: goal

        # 4-channel: current location
        row = self.s // self.ncol
        col = self.s % self.ncol
        box_obs[3][row][col] = 1

        return box_obs
