import random
import numpy as np
from icecream import ic
from torch.distributions import Normal

from codes.a_config._rl_parameters.off_policy.parameter_td3 import TD3ActionType
from codes.d_agents.actions import ContinuousActionSelector


class ContinuousNormalSACActionSelector(ContinuousActionSelector):
    def __call__(self, mu_v, logstd_v):
        dist = Normal(loc=mu_v, scale=logstd_v)
        actions = dist.sample().data.cpu().numpy()

        actions = np.clip(actions, -1.0, 1.0)
        return actions


class SomeTimesBlowSACActionSelector(ContinuousNormalSACActionSelector):
    def __init__(
            self, mu_v, logstd_v,
            blowing_action_rate=0.0002, min_blowing_action=-1.0, max_blowing_action=1.0, params=None
    ):
        super(SomeTimesBlowSACActionSelector, self).__init__()
        self.blowing_action_rate = blowing_action_rate
        self.min_blowing_action = min_blowing_action
        self.max_blowing_action = max_blowing_action
        self.time_steps = 0
        self.next_time_steps_of_random_blowing_action = int(random.expovariate(self.blowing_action_rate))

    def __call__(self,  mu_v, logstd_v): #default ou_sigma = 0.2
        if self.time_steps == 0:
            print("next_time_steps_of_random_blowing_action: {0}".format(
                self.next_time_steps_of_random_blowing_action
            ))

        self.time_steps += 1

        dist = Normal(loc=mu_v, scale=logstd_v)
        actions = dist.sample().data.cpu().numpy()

        if self.time_steps >= self.next_time_steps_of_random_blowing_action:
            actions += np.random.uniform(
                low=self.min_blowing_action, high=self.max_blowing_action, size=actions.shape
            )

            self.next_time_steps_of_random_blowing_action = self.time_steps + int(random.expovariate(self.blowing_action_rate))
            print("Internal Blowing Action: {0}, next_time_steps_of_random_blowing_action: {1}".format(
                actions,
                self.next_time_steps_of_random_blowing_action
            ))
        else:
            actions = np.clip(actions, -1.0, 1.0)

        return actions