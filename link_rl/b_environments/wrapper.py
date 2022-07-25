import warnings
from abc import abstractmethod

import numpy as np
import gym
from gym import spaces


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

    @abstractmethod
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


class ReverseActionWrapper(gym.Wrapper):
    def __init__(self, env, threshold):
        super().__init__(env)
        self.t = 0
        if threshold < 0:
            ValueError("Must be threshold >= 0")
        else:
            self.threshold = threshold

    def reset(self, **kwargs):
        self.t = 0
        return super().reset(**kwargs)

    def step(self, action):
        current = self.t
        self.t += 1

        quotient = current // self.threshold
        if quotient % 2 == 0:
            return self.env.step(self.action(action))
        else:
            return self.env.step(self.reverse_action(action))

        # if current < self.threshold:
        #     return self.env.step(self.action(action))
        # else:
        #     return self.env.step(self.reverse_action(action))

    def action(self, action):
        return action

    @abstractmethod
    def reverse_action(self, action):
        raise NotImplementedError


class ReverseActionCartpole(ReverseActionWrapper):
    def __init__(self, env, threshold=30):
        super().__init__(env=env, threshold=threshold)

    def reverse_action(self, action):
        return 1 - action


class CartpoleWithoutVelocity(gym.ObservationWrapper):
    def __init__(self, env):
        # | Num   | Observation             | Min                    | Max                  |
        # | ----- | ----------------------- | ---------------------- | -------------------- |
        # | 0     | CartPosition            | -4.8                   | 4.8                  |   O
        # | 1     | CartVelocity            | -Inf                   | Inf                  |   X
        # | 2     | PoleAngle               | ~ -0.418 rad(-24°)     | ~ 0.418rad(24°)      |   O
        # | 3     | PoleAngularVelocity     | -Inf                   | Inf                  |   X
        super().__init__(env)
        high = np.array(
            [
                self.x_threshold * 2,
                self.theta_threshold_radians * 2,
            ],
            dtype=np.float32,
        )
        self.observation_space = gym.spaces.Box(-high, high, dtype=np.float32)

    def observation(self, observation):
        cart_position, _, pole_angle, _ = observation
        return cart_position, pole_angle


class LunarLanderWithoutVelocity(gym.Wrapper):
    def __init__(self, env):
        # | Num   | Observation             | Min                    | Max                  |
        # | ----- | ----------------------- | ---------------------- | -------------------- |
        # | 0     | pos.x                   | -Inf                   | Inf                  |   O
        # | 1     | pos.y                   | -Inf                   | Inf                  |   O
        # | 2     | vel.x                   | -Inf                   | Inf                  |   X
        # | 3     | vel.y                   | -Inf                   | Inf                  |   X
        # | 4     | angle                   | -Inf                   | Inf                  |   O
        # | 5     | angular velocity        | -Inf                   | Inf                  |   X
        # | 6     | legs[0].ground_contact  | -Inf                   | Inf                  |   O
        # | 7     | legs[1].ground_contact  | -Inf                   | Inf                  |   O
        super().__init__(env)

        self.observation_space = gym.spaces.Box(
            -np.inf, np.inf, shape=(5,), dtype=np.float32
        )

        self.episode_reward = 0

    def reset(self, **kwargs):
        self.episode_reward = 0

        if kwargs.get("return_info", False):
            obs, info = self.env.reset(**kwargs)
            return self.observation(obs), info
        else:
            return self.observation(self.env.reset(**kwargs))

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        self.episode_reward += reward
        return self.observation(observation), reward, self.done(done), info

    def observation(self, observation):
        pos_x, pos_y, _, _, angle, _, legs_0_contact, legs_1_contact = observation
        return pos_x, pos_y, angle, legs_0_contact, legs_1_contact

    def done(self, done):
        if self.episode_reward < -300:
            return True
        return done


class AcrobotWithoutVelocity(gym.ObservationWrapper):
    def __init__(self, env):
        # | Num   | Observation                | Min                 | Max               |
        # | ----- | -------------------------- | ------------------- | ----------------- |
        # | 0     | Cosine of theta1           | -1                  | 1                 | O
        # | 1     | Sine of theta1             | -1                  | 1                 | O
        # | 2     | Cosine of theta2           | -1                  | 1                 | O
        # | 3     | Sine of theta2             | -1                  | 1                 | O
        # | 4     | Angular velocity of theta1 | ~ -12.567 (-4 * pi) | ~ 12.567 (4 * pi) | X
        # | 5     | Angular velocity of theta2 | ~ -28.274 (-9 * pi) | ~ 28.274 (9 * pi) | X
        super().__init__(env)

        self.observation_space = gym.spaces.Box(
            -1, 1, shape=(4,), dtype=np.float32
        )

    def observation(self, observation):
        cos1, sin1, cos2, sin2, _, _ = observation
        return cos1, sin1, cos2, sin2


class MountainCarWithoutVelocity(gym.ObservationWrapper):
    def __init__(self, env):
        # | Num   | Observation                | Min        | Max        |
        # | ----- | -------------------------- | ---------- | ---------- |
        # | 0     | Cosine of theta1           | -Inf       | Inf        | O
        # | 1     | Sine of theta1             | -Inf       | Inf        | X
        super().__init__(env)

        self.observation_space = gym.spaces.Box(
            self.min_position, self.max_position, shape=(1,), dtype=np.float32
        )

    def observation(self, observation):
        position, velocity = observation
        return position,


class CarRacingObservationTransposeWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

        STATE_W = 96
        STATE_H = 96

        self.observation_space = spaces.Box(
            low=0, high=255, shape=(3, STATE_H, STATE_W), dtype=np.uint8
        )

    def observation(self, observation):
        return np.transpose(observation, axes=(2, 1, 0))


class FrameStackVectorizedEnvWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

    def observation(self, observation):
        observation = np.swapaxes(observation, 0, 1)
        return observation


class ReturnInfoEnvWrapper(gym.Wrapper):
    def reset(self, return_info=False):
        observation = self.env.reset()
        info = None
        return observation, info


class EvoGymActionMinusOneWrapper(gym.Wrapper):
    def step(self, action):
        observation, reward, done, info = self.env.step(action - 1)
        return observation, reward, done, info
