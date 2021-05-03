import torch
from torch.distributions import Categorical, Normal
import numpy as np

from codes.d_agents.actions import ActionSelector, ContinuousActionSelector


class DiscreteCategoricalActionSelector(ActionSelector):
    """
    Converts probabilities of actions into action by sampling them
    """
    def __call__(self, probs):
        dist = Categorical(probs=probs)
        actions = dist.sample().cpu().detach().numpy()
        return np.array(actions)


class ContinuousNormalActionSelector(ContinuousActionSelector):
    def __call__(self, mu_v, var_v):
        # covariance_matrix = torch.diag_embed(var_v)
        # dist = MultivariateNormal(loc=mu_v, covariance_matrix=covariance_matrix)

        dist = Normal(loc=mu_v, scale=torch.sqrt(var_v))
        actions = dist.sample().data.cpu().numpy()
        actions = np.clip(actions, -1.0, 1.0)
        return actions


# class ContinuousNormalActionSelector(ContinuousActionSelector):
#     def __call__(self, mu_v, logstd_v, action_min, action_max):
#         mu = mu_v.data.cpu().numpy()
#         logstd = logstd_v.data.cpu().numpy()
#         rnd = np.random.normal(size=logstd.shape)
#         actions = mu + np.exp(logstd) * rnd
#         actions = np.clip(actions, action_min, action_max)
#         return actions