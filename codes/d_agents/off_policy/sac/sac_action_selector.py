import random
import numpy as np
import torch
from icecream import ic
from torch.distributions import Normal
import torch.nn.functional as F

from codes.a_config._rl_parameters.off_policy.parameter_td3 import TD3ActionType
from codes.d_agents.actions import ContinuousActionSelector


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

    def select_reparameterization_trick_action(self, mu_v, logstd_v):
        assert logstd_v is not None

        dist = Normal(loc=mu_v, scale=torch.exp(logstd_v))
        actions_v = dist.rsample()  # for reparameterization trick (mean + std * N(0,1))
        actions = F.tanh(actions_v)

        entropy_v = dist.entropy()

        return actions, entropy_v

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