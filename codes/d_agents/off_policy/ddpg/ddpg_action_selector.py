import random
import numpy as np

from codes.a_config._rl_parameters.off_policy.parameter_ddpg import DDPGActionType
from codes.d_agents.actions import ActionSelector


class DDPGActionSelector:
    def __init__(self, noise_enabled, ou_mu=None, ou_theta=0.15, ou_dt=0.01, ou_sigma=2.0, epsilon=0.0, params=None):
        self.noise_enabled = noise_enabled
        self.ou_mu = ou_mu
        self.ou_theta = ou_theta
        self.ou_dt = ou_dt
        self.ou_sigma = ou_sigma
        self.epsilon = epsilon
        self.params = params

    def action_select(self, actions, noises, global_uncertainty):
        if isinstance(noises, list):
            noises = np.asarray(noises)

        if noises.ndim == 1:
            noises = np.expand_dims(noises, axis=-1)

        if self.params.TYPE_OF_DDPG_ACTION == DDPGActionType.GAUSSIAN_NOISE_WITH_EPSILON:
            if self.noise_enabled and self.epsilon > 0.0:
                noises = np.random.normal(size=actions.shape, loc=0, scale=1.0)
                noises = self.epsilon * noises
            else:
                noises = np.zeros_like(actions)
        elif self.params.TYPE_OF_DDPG_ACTION == DDPGActionType.OU_NOISE_WITH_EPSILON:
            if self.noise_enabled and self.epsilon > 0.0:
                # agent_states = 1.0       +    0.15 * (0.0 - 1.0)            + new_normal_random
                noises = noises + self.ou_theta * (self.ou_mu - noises) + self.ou_sigma * np.random.normal(size=noises.shape)
                noises = self.epsilon * noises
            else:
                noises = np.zeros_like(actions)
        elif self.params.TYPE_OF_DDPG_ACTION == DDPGActionType.UNCERTAINTY:
            if self.noise_enabled:
                if random.random() < global_uncertainty:
                    noises = noises + self.ou_theta * (self.ou_mu - noises) * self.ou_dt + \
                             self.ou_sigma * np.sqrt(self.ou_dt) * np.random.normal(size=noises.shape)
                else:
                    noises = np.zeros_like(actions)
            else:
                noises = np.zeros_like(actions)
        elif self.params.TYPE_OF_DDPG_ACTION == DDPGActionType.ONLY_OU_NOISE:
            if self.noise_enabled:
                noises = noises + self.ou_theta * (self.ou_mu - noises) + self.ou_sigma * np.random.normal(
                    size=noises.shape)
            else:
                noises = np.zeros_like(actions)
        elif self.params.TYPE_OF_DDPG_ACTION == DDPGActionType.ONLY_GREEDY:
            noises = np.zeros_like(actions)
        else:
            raise ValueError()

        actions = actions + noises
        actions = np.clip(actions, -1.0, 1.0)

        return actions, noises

    def __call__(self, mu, noises, global_uncertainty=1.0):
        assert isinstance(mu, np.ndarray)
        actions = np.copy(mu)

        actions, noises = self.action_select(actions, noises, global_uncertainty)

        return actions, noises


class SomeTimesBlowDDPGActionSelector(DDPGActionSelector):
    def __init__(
            self, noise_enabled, ou_mu, ou_theta=0.15, ou_dt=0.01, ou_sigma=0.2,
            blowing_action_rate=0.0002, min_blowing_action=-1.0, max_blowing_action=1.0, epsilon=0.0, params=None
    ):
        super(SomeTimesBlowDDPGActionSelector, self).__init__(
            noise_enabled, ou_mu, ou_theta, ou_dt, ou_sigma, epsilon, params=params
        )
        self.blowing_action_rate = blowing_action_rate
        self.min_blowing_action = min_blowing_action
        self.max_blowing_action = max_blowing_action
        self.time_steps = 0
        self.next_time_steps_of_random_blowing_action = int(random.expovariate(self.blowing_action_rate))

    def __call__(self, mu, noises, global_uncertainty=1.0): #default ou_sigma = 0.2
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
            actions, noises = self.action_select(actions, noises, global_uncertainty)

        return actions, noises


class EpsilonGreedyD4PGActionSelector(ActionSelector):
    def __init__(self, epsilon=0.05):
        self.epsilon = epsilon

    def __call__(self, actions):
        assert isinstance(actions, np.ndarray)
        actions += self.epsilon * np.random.normal(size=actions.shape)
        return actions
