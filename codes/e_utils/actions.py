import random

import numpy as np
from typing import Union

import torch


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
    def __init__(self, epsilon=0.05, action_selector=None):
        self.epsilon = epsilon
        self.action_selector = action_selector if action_selector is not None else ArgmaxActionSelector()

    def __call__(self, scores):
        assert isinstance(scores, np.ndarray)
        batch_size, n_actions = scores.shape
        actions = self.action_selector(scores)
        mask = np.random.random(size=batch_size) < self.epsilon
        rand_actions = np.random.choice(n_actions, sum(mask))
        actions[mask] = rand_actions
        return actions


class EpsilonGreedySomeTimesBlowDQNActionSelector(ActionSelector):
    def __init__(
            self, epsilon=0.05, blowing_action_rate=0.0002,
            min_blowing_action_idx=-1.0, max_blowing_action_idx=1.0, action_selector=None
    ):
        self.epsilon = epsilon
        self.action_selector = action_selector if action_selector is not None else ArgmaxActionSelector()

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
        actions = self.action_selector(scores)

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

        if self.ou_enabled and self.epsilon > 0.0:
            # agent_states = 1.0       +    0.15 * (0.0 - 1.0)            + new_normal_random
            agent_states = agent_states + ou_theta * (ou_mu - agent_states) + ou_sigma * np.random.normal(size=actions.shape)
            actions = actions + self.epsilon * agent_states

        return actions, agent_states


class EpsilonGreedyD4PGActionSelector(ActionSelector):
    def __init__(self, epsilon=0.05):
        self.epsilon = epsilon

    def __call__(self, actions):
        assert isinstance(actions, np.ndarray)
        actions += self.epsilon * np.random.normal(size=actions.shape)
        return actions


class ProbabilityActionSelector(ActionSelector):
    """
    Converts probabilities of actions into action by sampling them
    """
    def __call__(self, probs):
        assert isinstance(probs, np.ndarray)
        actions = []
        for prob in probs:
            actions.append(np.random.choice(len(prob), p=prob))
        return np.array(actions)


class ContinuousNormalActionSelector(ContinuousActionSelector):
    def __call__(self, mu_v, logstd_v, action_min, action_max):
        mu = mu_v.data.cpu().numpy()
        logstd = logstd_v.data.cpu().numpy()
        rnd = np.random.normal(size=logstd.shape)
        actions = mu + np.exp(logstd) * rnd
        actions = np.clip(actions, action_min, action_max)
        return actions


# class ContinuousNormalActionSelector(ContinuousActionSelector):
#     def __call__(self, mu_v, var_v, action_min, action_max):
#         mu = mu_v.data.cpu().numpy()
#         sigma = torch.sqrt(var_v).data.cpu().numpy()
#         actions = np.random.normal(mu, sigma)
#         actions = np.clip(actions, action_min, action_max)
#         return actions


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
        self.udpate(0)

    def udpate(self, frame: int):
        eps = self.eps_start - frame / self.eps_frames
        self.action_selector.epsilon = max(self.eps_final, eps)