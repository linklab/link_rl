import random

import torch
from torch.distributions import Categorical, Normal, MultivariateNormal
import numpy as np

from codes.d_agents.actions import ActionSelector, ContinuousActionSelector, DiscreteActionSelector
from codes.e_utils.names import AgentMode


class DiscreteCategoricalActionSelector(DiscreteActionSelector):
    """
    Converts probabilities of actions into action by sampling them
    """
    def __init__(self, params):
        self.params = params

    def __call__(self, probs, agent_mode):
        with torch.no_grad():
            if agent_mode == AgentMode.TRAIN:
                dist = Categorical(probs=probs)
                actions = dist.sample().cpu().detach().numpy()
            else:
                actions = torch.argmax(probs, dim=-1, keepdim=True).squeeze(dim=-1).cpu().detach().numpy()

        return np.array(actions)


class ContinuousNormalActionSelector(ContinuousActionSelector):
    def __init__(self, params):
        self.params = params

    def select_action(self, mu_v, logstd_v):
        with torch.no_grad():
            if logstd_v is not None:
                dist = Normal(loc=mu_v, scale=torch.exp(logstd_v))
                actions = dist.sample().data.cpu().numpy()
            else:
                actions = mu_v.data.cpu().numpy()

        actions = np.clip(actions, -1.0, 1.0)

        return actions

    def __call__(self, mu_v, logstd_v=None):
        return self.select_action(mu_v, logstd_v)


class SomeTimesBlowContinuousNormalActionSelector(ContinuousNormalActionSelector):
    def __init__(
            self, blowing_action_rate=0.0002, min_blowing_action=-1.0, max_blowing_action=1.0, params=None
    ):
        super(SomeTimesBlowContinuousNormalActionSelector, self).__init__(params=params)
        self.blowing_action_rate = blowing_action_rate
        self.min_blowing_action = min_blowing_action
        self.max_blowing_action = max_blowing_action
        self.time_steps = 0
        self.next_time_steps_of_random_blowing_action = int(random.expovariate(self.blowing_action_rate))

    def __call__(self,  mu_v, logstd_v=None):
        if self.time_steps == 0:
            print("next_time_steps_of_random_blowing_action: {0}".format(
                self.next_time_steps_of_random_blowing_action
            ))

        self.time_steps += 1

        actions = self.select_action(mu_v, logstd_v)

        if self.time_steps >= self.next_time_steps_of_random_blowing_action:
            actions += np.random.uniform(
                low=self.min_blowing_action, high=self.max_blowing_action, size=actions.shape
            )

            self.next_time_steps_of_random_blowing_action = self.time_steps + int(random.expovariate(self.blowing_action_rate))
            print("[{0:6}/{1}] Internal Blowing Action: {2}, next_time_steps_of_random_blowing_action: {3}".format(
                self.time_steps,
                self.params.MAX_GLOBAL_STEP,
                actions,
                self.next_time_steps_of_random_blowing_action
            ))

        return actions