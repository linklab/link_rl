import random

import numpy as np
from typing import Union

import torch
from icecream import ic
from torch.distributions import MultivariateNormal, Normal, Categorical

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
from codes.a_config.parameters import PARAMETERS as params


class ActionSelector:
    """
    Abstract class which converts scores to the actions
    """
    def __call__(self, scores):
        raise NotImplementedError


class ContinuousActionSelector:
    """
    Abstract class which converts scores to the actions
    """
    def __call__(self, mu_v, var_v, action_min, action_max):
        raise NotImplementedError


class ArgmaxActionSelector(ActionSelector):
    """
    Selects actions using argmax
    """
    def __init__(self, supports=None):
        super(ArgmaxActionSelector).__init__()
        self.supports = supports

    def __call__(self, q_values):
        assert isinstance(q_values, np.ndarray)

        if params.DISTRIBUTIONAL:
            dist = q_values * self.supports
            action = dist.sum(2).max(1)[1].numpy()[0]
        else:
            action = np.argmax(q_values, axis=1)
        return action


class EpsilonGreedyDQNActionSelector(ActionSelector):
    def __init__(self, epsilon=0.05, default_action_selector=None):
        self.epsilon = epsilon
        self.default_action_selector = default_action_selector if default_action_selector is not None else ArgmaxActionSelector()

    def __call__(self, q_values):
        assert isinstance(q_values, np.ndarray)
        batch_size, n_actions = q_values.shape
        actions = self.default_action_selector(q_values)
        mask = np.random.random(size=batch_size) < self.epsilon
        rand_actions = np.random.choice(n_actions, sum(mask))
        actions[mask] = rand_actions
        return actions


class EpsilonGreedySomeTimesBlowDQNActionSelector(ActionSelector):
    #TODO: max_blowing_action_idx
    def __init__(
            self, epsilon=0.05, blowing_action_rate=0.0002,
            min_blowing_action_idx=0, max_blowing_action_idx=-1, default_action_selector=None
    ):
        self.epsilon = epsilon
        self.default_action_selector = default_action_selector if default_action_selector is not None else ArgmaxActionSelector()

        self.blowing_action_rate = blowing_action_rate
        self.min_blowing_action_idx = min_blowing_action_idx
        self.max_blowing_action_idx = max_blowing_action_idx
        self.time_steps = 0
        self.next_time_steps_of_random_blowing_action = int(random.expovariate(self.blowing_action_rate))

    def __call__(self, scores):
        assert isinstance(scores, np.ndarray)
        if self.time_steps == 0:
            print("next_time_steps_of_random_blowing_action: {0}".format(
                self.next_time_steps_of_random_blowing_action
            ))

        self.time_steps += 1
        batch_size, n_actions = scores.shape
        actions = self.default_action_selector(scores)

        if self.time_steps >= self.next_time_steps_of_random_blowing_action:
            actions = np.random.choice(
                a=[self.min_blowing_action_idx, self.max_blowing_action_idx], size=actions.shape
            )

            # actions += np.random.uniform(
            #     low=self.min_blowing_action, high=self.max_blowing_action, size=actions.shape
            # )

            self.next_time_steps_of_random_blowing_action = self.time_steps + int(random.expovariate(self.blowing_action_rate))
            print("Internal Blowing Action: {0}, next_time_steps_of_random_blowing_action: {1}".format(
                actions,
                self.next_time_steps_of_random_blowing_action
            ))
        else:
            mask = np.random.random(size=batch_size) < self.epsilon
            rand_actions = np.random.choice(n_actions, sum(mask))
            actions[mask] = rand_actions
        return actions


class DDPGActionSelector:
    def __init__(self, ou_enabled, ou_mu=None, ou_theta=0.15, ou_dt=0.01, ou_sigma=2.0, epsilon=0.0):
        self.ou_enabled = ou_enabled
        self.ou_mu = ou_mu
        self.ou_theta = ou_theta
        self.ou_dt = ou_dt
        self.ou_sigma = ou_sigma
        self.epsilon = epsilon

    def action_select(self, actions, noises, global_uncertainty):
        if isinstance(noises, list):
            noises = np.asarray(noises)

        if noises.ndim == 1:
            noises = np.expand_dims(noises, axis=-1)

        if params.TYPE_OF_ACTION == "old":
            if self.ou_enabled and self.epsilon > 0.0:
                # agent_states = 1.0       +    0.15 * (0.0 - 1.0)            + new_normal_random
                noises = noises + self.ou_theta * (self.ou_mu - noises) + self.ou_sigma * np.random.normal(
                    size=noises.shape)
                actions = actions + self.epsilon * noises
            else:
                noises = np.zeros_like(actions)
        elif params.TYPE_OF_ACTION == "current":
            if self.ou_enabled:
                if random.random() < global_uncertainty:
                    noises = noises + self.ou_theta * (self.ou_mu - noises) * self.ou_dt + \
                             self.ou_sigma * np.sqrt(self.ou_dt) * np.random.normal(size=noises.shape)
                else:
                    noises = np.zeros_like(actions)

                actions = actions + noises
            else:
                noises = np.zeros_like(actions)
        else:
            raise ValueError()

        return actions, noises

    def __call__(self, mu, noises, global_uncertainty=1.0):
        assert isinstance(mu, np.ndarray)
        actions = np.copy(mu)

        actions, noises = self.action_select(actions, noises, global_uncertainty)

        return actions, noises


class SomeTimesBlowDDPGActionSelector(DDPGActionSelector):
    def __init__(
            self, ou_enabled, ou_mu, ou_theta=0.15, ou_dt=0.01, ou_sigma=0.2,
            blowing_action_rate=0.0002, min_blowing_action=-1.0, max_blowing_action=1.0, epsilon=0.0
    ):
        super(SomeTimesBlowDDPGActionSelector, self).__init__(ou_enabled, ou_mu, ou_theta, ou_dt, ou_sigma, epsilon)
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

# class DDPGActionSelector:
#     def __init__(self, ou_enabled, ou_mu=None, ou_theta=0.15, ou_dt=0.01, ou_sigma=2.0):
#         self.ou_enabled = ou_enabled
#         self.ou_mu = ou_mu
#         self.ou_theta = ou_theta
#         self.ou_dt = ou_dt
#         self.ou_sigma = ou_sigma
#
#     def __call__(self, mu, noises, global_uncertainty=1.0):
#         assert isinstance(mu, np.ndarray)
#         actions = np.copy(mu)
#
#         if self.ou_enabled:
#             if isinstance(noises, list):
#                 noises = np.asarray(noises)
#
#             # print(noises.shape, "#####")
#             # if noises.ndim == 1:
#             #     noises = np.expand_dims(noises, axis=-1)
#
#             noises = noises + self.ou_theta * (self.ou_mu - noises) * self.ou_dt + \
#                      self.ou_sigma * np.sqrt(self.ou_dt) * np.random.normal(size=noises.shape)
#
#             noises = global_uncertainty * noises
#
#             actions = actions + noises
#             # print("actions: {0:7.4f}, global_uncertainty: {1:7.4f}, noises: {2:7.4f}".format(
#             #     actions[0][0], global_uncertainty, noises[0][0]
#             # ))
#         else:
#             noises = np.zeros_like(actions)
#             # print("actions: {0:7.4f} - ou_enabled: False".format(actions[0][0]))
#
#         # print("mu : {0:2.4f}, action : {1:2.4f}".format(mu[0][0], actions[0][0]))
#         return actions, noises


class EpsilonGreedyD4PGActionSelector(ActionSelector):
    def __init__(self, epsilon=0.05):
        self.epsilon = epsilon

    def __call__(self, actions):
        assert isinstance(actions, np.ndarray)
        actions += self.epsilon * np.random.normal(size=actions.shape)
        return actions


class DiscreteCategoricalActionSelector(ActionSelector):
    """
    Converts probabilities of actions into action by sampling them
    """
    def __call__(self, probs):
        dist = Categorical(probs=probs)
        actions = dist.sample().cpu().detach().numpy()
        return np.array(actions)


# class ContinuousNormalActionSelector(ContinuousActionSelector):
#     def __call__(self, mu_v, logstd_v, action_min, action_max):
#         mu = mu_v.data.cpu().numpy()
#         logstd = logstd_v.data.cpu().numpy()
#         rnd = np.random.normal(size=logstd.shape)
#         actions = mu + np.exp(logstd) * rnd
#         actions = np.clip(actions, action_min, action_max)
#         return actions


class ContinuousNormalActionSelector(ContinuousActionSelector):
    def __call__(self, mu_v, var_v, action_min, action_max):
        # covariance_matrix = torch.diag_embed(var_v)
        # dist = MultivariateNormal(loc=mu_v, covariance_matrix=covariance_matrix)

        dist = Normal(loc=mu_v, scale=torch.sqrt(var_v))
        actions = dist.sample().data.cpu().numpy()
        actions = np.clip(actions, action_min, action_max)
        return actions


class TD3ActionSelector:
    def __init__(self, epsilon, act_noise=0.0, noise_clip=0.0):
        self.act_noise = act_noise
        self.noise_clip = noise_clip
        self.epsilon = epsilon

    def __call__(self, mu, noises=None):
        assert isinstance(mu, np.ndarray)
        actions = np.copy(mu)
        if self.act_noise == 0.0:
            noises = np.zeros_like(shape=actions.shape)
        else:
            noises = np.random.normal(size=mu.shape, loc=0, scale=self.act_noise)

        if params.TYPE_OF_ACTION == "old":
            actions = actions + self.epsilon*noises
        else:
            actions = actions + noises

        #ic(noises)

        return actions, noises

class EpsilonGreedySomeTimesBlowDDPGActionSelector:
    def __init__(
            self, epsilon, ou_enabled, scale_factor,
            blowing_action_rate=0.0002, min_blowing_action=-1.0, max_blowing_action=1.0
    ):
        self.epsilon = epsilon
        self.ou_enabled = ou_enabled
        self.scale_factor = scale_factor

        self.blowing_action_rate = blowing_action_rate
        self.min_blowing_action = min_blowing_action
        self.max_blowing_action = max_blowing_action
        self.time_steps = 0
        self.next_time_steps_of_random_blowing_action = int(random.expovariate(self.blowing_action_rate))

    def __call__(self, mu, agent_states, ou_rho=0.15, ou_mu=0.0, ou_dt=0.1, ou_sigma=0.2): #default ou_sigma = 0.2
        assert isinstance(mu, np.ndarray)
        if self.time_steps == 0:
            print("next_time_steps_of_random_blowing_action: {0}".format(
                self.next_time_steps_of_random_blowing_action
            ))

        self.time_steps += 1
        actions = np.copy(mu)

        if isinstance(agent_states, list):
            agent_states = np.asarray(agent_states)

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
            if self.ou_enabled:
                # agent_states = 1.0       +    0.15 * (0.0 - 1.0)            + new_random
                agent_states = agent_states + ou_rho * (ou_mu - agent_states) + ou_sigma * np.sqrt(ou_dt) * np.random.normal(size=actions.shape)

                noises = self.epsilon * agent_states
                actions = actions + noises
            else:
                noises = np.zeros_like(actions)

        new_agent_states = noises

        return actions, new_agent_states


class EpsilonGreedyDDPGActionSelector:
    def __init__(self, epsilon, ou_enabled, scale_factor):
        self.epsilon = epsilon
        self.ou_enabled = ou_enabled
        self.scale_factor = scale_factor

    def __call__(self, mu, agent_states, ou_theta=0.15, ou_mu=0.0, ou_sigma=0.2): #default ou_sigma = 0.2
        assert isinstance(mu, np.ndarray)
        actions = np.copy(mu)
        if isinstance(agent_states, list):
            agent_states = np.asarray(agent_states)

        #ic(agent_states)

        if agent_states.ndim == 1:
            agent_states = np.expand_dims(agent_states, axis=-1)

        if self.ou_enabled and self.epsilon > 0.0:
            # agent_states = 1.0       +    0.15 * (0.0 - 1.0)            + new_normal_random
            agent_states = agent_states + ou_theta * (ou_mu - agent_states) + ou_sigma * np.random.normal(size=agent_states.shape)
            actions = actions + self.epsilon * agent_states

        return actions, agent_states


class EpsilonTracker:
    """
    Updates epsilon according to linear schedule
    """
    def __init__(
        self, action_selector, eps_start: Union[int, float], eps_final: Union[int, float], eps_frames: int
    ):
        self.action_selector = action_selector
        self.eps_start = eps_start
        self.eps_final = eps_final
        self.eps_frames = eps_frames
        #self.udpate(0)

    def udpate(self, frame: int):
        eps = self.eps_start - (frame / self.eps_frames)
        self.action_selector.epsilon = max(self.eps_final, eps)
