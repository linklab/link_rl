import random

import numpy as np
from typing import Union

import torch
from icecream import ic
from torch.distributions import MultivariateNormal, Normal, Categorical

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


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
    def __call__(self, scores):
        assert isinstance(scores, np.ndarray)
        return np.argmax(scores, axis=1)


class EpsilonGreedyDQNActionSelector(ActionSelector):
    def __init__(self, epsilon=0.05, default_action_selector=None):
        self.epsilon = epsilon
        self.default_action_selector = default_action_selector if default_action_selector is not None else ArgmaxActionSelector()

    def __call__(self, scores):
        assert isinstance(scores, np.ndarray)
        batch_size, n_actions = scores.shape
        actions = self.default_action_selector(scores)
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


class SomeTimesBlowDDPGActionSelector:
    def __init__(
            self, ou_enabled, ou_theta=0.15, ou_dt=0.01, ou_sigma=0.2,
            blowing_action_rate=0.0002, min_blowing_action=-1.0, max_blowing_action=1.0
    ):
        self.ou_enabled = ou_enabled
        self.ou_theta = ou_theta
        self.ou_dt = ou_dt
        self.ou_sigma = ou_sigma

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
            if self.ou_enabled:
                if isinstance(noises, list):
                    noises = np.asarray(noises)

                if noises.ndim == 1:
                    noises = np.expand_dims(noises, axis=-1)

                noises = noises + global_uncertainty * (
                        self.ou_theta * (actions - noises) * self.ou_dt +
                        self.ou_sigma * np.sqrt(self.ou_dt) * np.random.normal(size=noises.shape)
                )

                actions = actions + noises
            else:
                noises = np.zeros_like(actions)

        return actions, noises


class DDPGActionSelector:
    def __init__(self, ou_enabled, ou_theta=0.15, ou_dt=0.01, ou_sigma=2.0):
        self.ou_enabled = ou_enabled
        self.ou_theta = ou_theta
        self.ou_dt = ou_dt
        self.ou_sigma = ou_sigma

    def __call__(self, mu, noises, global_uncertainty=1.0):
        assert isinstance(mu, np.ndarray)
        actions = np.copy(mu)

        if self.ou_enabled:
            if isinstance(noises, list):
                noises = np.asarray(noises)

            if noises.ndim == 1:
                noises = np.expand_dims(noises, axis=-1)

            noises = noises + global_uncertainty * (
                    self.ou_theta * (actions - noises) * self.ou_dt +
                    self.ou_sigma * np.sqrt(self.ou_dt) * np.random.normal(size=noises.shape)
            )

            actions = actions + noises
            # print("actions: {0:7.4f}, epsilon: {1:7.4f}, noises: {2:7.4f}".format(
            #     actions[0][0], self.epsilon, noises[0][0]
            # ))
        else:
            noises = np.zeros_like(actions)

        # print("mu : {0:2.4f}, action : {1:2.4f}".format(mu[0][0], actions[0][0]))
        return actions, noises


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
    def __init__(self, act_noise=0.0, noise_clip=0.0):
        self.act_noise = act_noise
        self.noise_clip = noise_clip

    def __call__(self, mu, noises=None):
        assert isinstance(mu, np.ndarray)
        actions = np.copy(mu)
        if self.act_noise == 0.0:
            noises = np.zeros_like(shape=actions.shape)
        else:
            noises = np.random.normal(size=mu.shape, loc=0, scale=self.act_noise)

        actions = actions + noises

        #ic(noises)

        return actions, noises


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
