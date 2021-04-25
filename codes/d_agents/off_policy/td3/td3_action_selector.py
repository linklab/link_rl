import random
import numpy as np
from icecream import ic

from codes.a_config._rl_parameters.off_policy.parameter_td3 import TD3ActionType


class TD3ActionSelector:
    def __init__(self, epsilon, act_noise=0.0, params=None):
        self.act_noise = act_noise
        self.epsilon = epsilon
        self.params = params

    def select_action(self, mu):
        actions = np.copy(mu)
        if self.act_noise == 0.0:
            noises = np.zeros(shape=actions.shape)
        else:
            noises = np.random.normal(size=mu.shape, loc=0, scale=self.act_noise)

        if self.params.TYPE_OF_TD3_ACTION == TD3ActionType.NORMAL_NOISE_WITH_EPSILON:
            actions = actions + self.epsilon * noises
        elif self.params.TYPE_OF_TD3_ACTION == TD3ActionType.NORMAL_NOISE:
            actions = actions + noises
        elif self.params.TYPE_OF_TD3_ACTION == TD3ActionType.ONLY_GREEDY:
            actions = actions
        else:
            raise ValueError()

        return actions, noises

    def __call__(self, mu, noises=None):
        assert isinstance(mu, np.ndarray)
        actions = np.copy(mu)
        actions, noises = self.select_action(actions)
        #ic(noises)

        return actions, noises


class SomeTimesBlowTD3ActionSelector(TD3ActionSelector):
    def __init__(
            self, act_noise=0.0,
            blowing_action_rate=0.0002, min_blowing_action=-1.0, max_blowing_action=1.0, epsilon=0.0, params=None
    ):
        super(SomeTimesBlowTD3ActionSelector, self).__init__(epsilon, act_noise=0.0, params=params)
        self.blowing_action_rate = blowing_action_rate
        self.min_blowing_action = min_blowing_action
        self.max_blowing_action = max_blowing_action
        self.time_steps = 0
        self.next_time_steps_of_random_blowing_action = int(random.expovariate(self.blowing_action_rate))
        self.act_noise = act_noise

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
            print("Internal Blowing Action: {0}, next_time_steps_of_random_blowing_action: {1}".format(
                actions,
                self.next_time_steps_of_random_blowing_action
            ))

            noises = np.zeros_like(actions)
        else:
            actions, noises = self.select_action(actions)

        return actions, noises