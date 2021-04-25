import numpy as np
from typing import Union

import torch
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
    def __init__(self, supports_numpy=None):
        super(ArgmaxActionSelector).__init__()
        self.supports_numpy = supports_numpy

    def __call__(self, q_values):
        assert isinstance(q_values, np.ndarray)

        if params.DISTRIBUTIONAL:
            # q_values.shape: (batch, 2, 51)
            # self.supports: (51,)
            dist = q_values * self.supports_numpy
            action = np.argmax(dist.sum(2), axis=1)
        else:
            action = np.argmax(q_values, axis=1)
        return action


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
