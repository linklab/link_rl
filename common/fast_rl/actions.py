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


class EpsilonGreedyActionSelector(ActionSelector):
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


class EpsilonGreedyD4PGActionSelector(ActionSelector):
    def __init__(self, epsilon=0.05, action_selector=None):
        self.epsilon = epsilon
        self.action_selector = action_selector if action_selector is not None else ArgmaxActionSelector()

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
    def __call__(self, mu_v, var_v, action_min, action_max):
        mu = mu_v.data.cpu().numpy()
        sigma = torch.sqrt(var_v).data.cpu().numpy()
        actions = np.random.normal(mu, sigma)
        actions = np.clip(actions, action_min, action_max)
        return actions


class EpsilonTracker:
    """
    Updates epsilon according to linear schedule
    """
    def __init__(
        self, action_selector: EpsilonGreedyActionSelector,
        eps_start: Union[int, float], eps_final: Union[int, float], eps_frames: int
    ):
        self.action_selector = action_selector
        self.eps_start = eps_start
        self.eps_final = eps_final
        self.eps_frames = eps_frames
        self.udpate(0)

    def udpate(self, frame: int):
        eps = self.eps_start - frame / self.eps_frames
        self.action_selector.epsilon = max(self.eps_final, eps)