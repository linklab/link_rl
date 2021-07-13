import torch
from torch.distributions import Categorical, Normal, MultivariateNormal
import numpy as np

from codes.d_agents.actions import ActionSelector, ContinuousActionSelector
from codes.e_utils.names import AgentMode


class DiscreteCategoricalActionSelector(ActionSelector):
    """
    Converts probabilities of actions into action by sampling them
    """
    def __init__(self, agent_mode):
        self.agent_mode = agent_mode

    def __call__(self, probs):
        with torch.no_grad():
            if self.agent_mode == AgentMode.TRAIN:
                dist = Categorical(probs=probs)
                actions = dist.sample().cpu().detach().numpy()
            else:
                actions = torch.argmax(probs, dim=-1, keepdim=True).squeeze(dim=-1).cpu().detach().numpy()

        return np.array(actions)


class ContinuousNormalActionSelector(ContinuousActionSelector):
    def __call__(self, mu_v, logstd_v=None):
        with torch.no_grad():
            if logstd_v is not None:
                dist = Normal(loc=mu_v, scale=torch.exp(logstd_v))
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