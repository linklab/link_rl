import random
from collections import deque

import numpy as np
from icecream import ic

from codes.a_config._rl_parameters.off_policy.parameter_td3 import TD3ActionType
from codes.e_utils.common_utils import ema
from codes.e_utils.names import EnvironmentName


class TD3ActionSelector:
    def __init__(self, epsilon, noise_std=0.0, params=None):
        self.noise_std = self.original_noise_std = noise_std
        self.epsilon = epsilon
        self.params = params
        # self.action_list = deque(maxlen=11)

    def select_action(self, mu):
        actions = np.copy(mu)

        # self.action_list.append(actions[0][0])
        #
        # actions = np.expand_dims(
        #     np.expand_dims(
        #         np.asarray(ema(list(self.action_list), len(self.action_list) - 1)[-1]),
        #         axis=-1
        #     ),
        #     axis=-1
        # )

        if self.noise_std == 0.0:
            noises = np.zeros(shape=actions.shape)
        else:
            noises = np.random.normal(size=mu.shape, loc=0, scale=self.noise_std)

        if self.params.TYPE_OF_TD3_ACTION == TD3ActionType.GAUSSIAN_NOISE_WITH_EPSILON:
            noises = self.epsilon * noises
            actions = actions + noises
        elif self.params.TYPE_OF_TD3_ACTION == TD3ActionType.GAUSSIAN_NOISE:
            actions = actions + noises
        elif self.params.TYPE_OF_TD3_ACTION == TD3ActionType.ONLY_GREEDY:
            actions = actions
        else:
            raise ValueError()

        actions = np.clip(actions, -1.0, 1.0)

        return actions, noises

    def __call__(self, mu, noises=None):
        assert isinstance(mu, np.ndarray)
        actions = np.copy(mu)
        actions, noises = self.select_action(actions)

        return actions, noises


class SomeTimesBlowTD3ActionSelector(TD3ActionSelector):
    def __init__(
            self, noise_std=0.0,
            blowing_action_rate=0.0002, min_blowing_action=-1.0, max_blowing_action=1.0, epsilon=0.0, params=None
    ):
        super(SomeTimesBlowTD3ActionSelector, self).__init__(epsilon, noise_std=0.0, params=params)

        if self.params.ENVIRONMENT_ID == EnvironmentName.REAL_DEVICE_DOUBLE_RIP:
            self.blowing_action_rate = 0.0002
        else:
            self.blowing_action_rate = 0.0002

        self.blowing_action_rate = blowing_action_rate
        self.min_blowing_action = min_blowing_action
        self.max_blowing_action = max_blowing_action
        self.time_steps = 0
        self.next_time_steps_of_random_blowing_action = int(random.expovariate(self.blowing_action_rate))
        self.noise_std = noise_std

    def __call__(self,  mu, noises=None): #default ou_sigma = 0.2
        assert isinstance(mu, np.ndarray)
        if self.time_steps == 0:
            print("next_time_steps_of_random_blowing_action: {0}".format(
                self.next_time_steps_of_random_blowing_action
            ))

        self.time_steps += 1
        actions = np.copy(mu)

        if self.time_steps >= self.next_time_steps_of_random_blowing_action:
            actions += np.random.uniform(
                low=self.min_blowing_action, high=self.max_blowing_action, size=actions.shape
            )

            self.next_time_steps_of_random_blowing_action = self.time_steps + int(random.expovariate(self.blowing_action_rate))
            print("[{0:6}/{1}] Internal Blowing Action: {2}, next_time_steps_of_random_blowing_action: {3}".format(
                self.time_steps,
                self.params.MAX_GLOBAL_STEP,
                actions,
                self.next_time_steps_of_random_blowing_action
            ))

            noises = np.zeros_like(actions)
        else:
            actions, noises = self.select_action(actions)

        return actions, noises