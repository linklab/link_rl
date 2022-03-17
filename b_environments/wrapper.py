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
        new_obs = np.zeros(self.discrete_observation_space_n)  # [0, 0, 0, 0]
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
        raise NotImplementedError()


class DiscreteToBox(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        assert isinstance(env.observation_space, gym.spaces.Discrete), \
            "Should only be used to wrap Discrete envs."

        self.discrete_observation_space_n = self.observation_space.n
        self.observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=(self.discrete_observation_space_n,)
        )

    def observation(self, observation):
        if observation is None:
            return None

        new_obs = np.zeros(self.discrete_observation_space_n)
        new_obs[observation] = 1

        return new_obs


class MakeBoxFrozenLake(gym.Wrapper):
    def __init__(self, random_map=True):
        if random_map:
            self._generate_random_map()
        else:
            env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=False)
            super().__init__(env)

        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=(4, self.nrow, self.ncol)
        )
        self.reward_range = (-100, 100)
        self.row = None
        self.col = None
        self.random_map = random_map

    def _generate_random_map(self):
        from gym.envs.toy_text.frozen_lake import generate_random_map

        random_map = generate_random_map(size=5, p=0.4)  # F:H = p:1-p
        env = gym.make("FrozenLake-v1", desc=random_map, is_slippery=False)

        super().__init__(env)

    def reset(self, return_info=False):
        if self.random_map:
            # print("맵 새로 생성")
            self._generate_random_map()

        if return_info:
            _, info = self.env.reset(return_info=return_info)
        else:
            _ = self.env.reset(return_info=return_info)
        self.row = self.s // self.ncol
        self.col = self.s % self.ncol

        if return_info:
            return self.observation(), info
        else:
            return self.observation()

    def step(self, action):
        _, _, done, info = self.env.step(action)
        self.row = self.s // self.ncol
        self.col = self.s % self.ncol

        return self.observation(), self.reward(), done, info

    def observation(self):
        box_obs = np.zeros(shape=(4, self.nrow, self.ncol), dtype=np.int)

        box_obs[0] = (self.desc == b'S')  # 1-channel: start
        box_obs[1] = (self.desc == b'H')  # 2-channel: hole
        box_obs[2] = (self.desc == b'G')  # 3-channel: goal
        box_obs[3][self.row][self.col] = 1  # 4-channel: current location

        return box_obs

    def reward(self):
        location = self.desc[self.row][self.col]

        if location == b'G':
            reward = 100
        elif location == b'H':
            reward = -1
        else:  # b'F' or b'S'
            reward = -1

        return reward


class ActionMaskWrapper(gym.Wrapper):
    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        return observation, reward, done, self.info(info)

    def reset(self, return_info=False):
        observation = self.env.reset()
        if return_info:
            # print("reset 호출 - info 있음")
            return observation, self.info()
        else:
            return observation

    def info(self, info=None):
        if info is None:
            info = dict()
        elif not isinstance(info, dict):
            TypeError("info is not dict")
        info["available_actions"], info["unavailable_actions"] = self.get_action_mask()
        return info

    def get_action_mask(self):
        raise NotImplementedError()


class FrozenLakeActionMask(ActionMaskWrapper):
    def get_action_mask(self):
        all_actions = list(range(4))  # Discrete(4)
        available_actions = []
        unavailable_actions = []

        for action in all_actions:  # about all actions
            next_row, next_col = self._next_state(self.row, self.col, action)
            if any([
                (next_row, next_col) == (self.row, self.col),  # toward the wall
                self.desc[next_row][next_col] == b'H'  # fall in a hole
            ]):
                action_is_possible = False
            else:
                action_is_possible = True
            available_actions.append(action_is_possible)
            unavailable_actions.append(not action_is_possible)

        return available_actions, unavailable_actions

    def _next_state(self, row, col, action):
        LEFT = 0
        DOWN = 1
        RIGHT = 2
        UP = 3
        if action == LEFT:
            col = max(col - 1, 0)
        elif action == DOWN:
            row = min(row + 1, self.nrow - 1)
        elif action == RIGHT:
            col = min(col + 1, self.ncol - 1)
        elif action == UP:
            row = max(row - 1, 0)

        return row, col


class VectorEnvReturnInfo(gym.vector.VectorEnvWrapper):
    def reset(self):
        obs, info = self.env.reset(return_info=True)
        return obs, info
