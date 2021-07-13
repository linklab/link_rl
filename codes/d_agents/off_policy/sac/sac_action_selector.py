import random
import numpy as np
import torch
from torch.distributions import Normal, Categorical

from codes.d_agents.actions import ContinuousActionSelector, ActionSelector
from codes.e_utils.names import AgentMode


class ContinuousNormalSACActionSelector(ContinuousActionSelector):
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


class SomeTimesBlowSACActionSelector(ContinuousNormalSACActionSelector):
    def __init__(
            self, blowing_action_rate=0.0002, min_blowing_action=-1.0, max_blowing_action=1.0, params=None
    ):
        super(SomeTimesBlowSACActionSelector, self).__init__(params=params)
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


class DiscreteCategoricalSACActionSelector(ActionSelector):
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
