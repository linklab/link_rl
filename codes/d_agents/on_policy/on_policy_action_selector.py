import torch
from torch.distributions import Categorical, Normal, MultivariateNormal
import numpy as np

from codes.d_agents.actions import ActionSelector, ContinuousActionSelector


class DiscreteCategoricalActionSelector(ActionSelector):
    """
    Converts probabilities of actions into action by sampling them
    """
    def __call__(self, probs):
        with torch.no_grad():
            dist = Categorical(probs=probs)
            actions = dist.sample().cpu().detach().numpy()

        return np.array(actions)


class ContinuousNormalActionSelector(ContinuousActionSelector):
    def __call__(self, mu_v, logstd_v=None):
        with torch.no_grad():
            if logstd_v is not None:
                dist = Normal(loc=mu_v, scale=logstd_v)
                actions = dist.sample().data.cpu().numpy()
            else:
                actions = mu_v.data.cpu().numpy()

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