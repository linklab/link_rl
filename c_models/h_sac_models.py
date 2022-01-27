from typing import Tuple
import torch
import numpy as np
from torch.distributions import Normal, TanhTransform, TransformedDistribution

from c_models.d_critic_models import DoubleQCriticModel
from c_models.c_actor_models import DiscreteActorModel, ContinuousStochasticActorModel
from e_main.config_single import config


class DiscreteSacModel:
    def __init__(
            self, observation_shape: Tuple[int], n_out_actions: int, n_discrete_actions=None, config=None,
            is_target_model=False
    ):
        self.config = config

        if is_target_model:
            self.actor_model = None
        else:
            self.actor_model = DiscreteActorModel(
                observation_shape=observation_shape, n_out_actions=n_out_actions, n_discrete_actions=n_discrete_actions,
                config=self.config
            ).to(self.config.DEVICE)

        self.critic_model = DoubleQCriticModel(
            observation_shape=observation_shape, n_out_actions=n_out_actions, n_discrete_actions=n_discrete_actions,
            config=self.config
        ).to(self.config.DEVICE)


class ContinuousSacModel:
    def __init__(
            self, observation_shape: Tuple[int], n_out_actions: int, config=None, is_target_model=False
    ):
        self.config = config

        if is_target_model:
            self.actor_model = None
        else:
            self.actor_model = ContinuousStochasticActorModel(
                observation_shape=observation_shape, n_out_actions=n_out_actions, config=config
            ).to(self.config.DEVICE)

        self.critic_model = DoubleQCriticModel(
            observation_shape=observation_shape, n_out_actions=n_out_actions, n_discrete_actions=None,
            config=self.config
        ) .to(self.config.DEVICE)

    def re_parameterization_trick_sample(self, obs):
        mu_v, sigma_v = self.actor_model.pi(obs)

        dist = Normal(loc=mu_v, scale=sigma_v)
        dist = TransformedDistribution(base_distribution=dist, transforms=TanhTransform(cache_size=1))

        action_v = dist.rsample()  # for reparameterization trick (mean + std * N(0,1))

        log_probs = dist.log_prob(action_v).sum(dim=-1, keepdim=True)
        # action_v.shape: [128, 1]
        # log_prob.shape: [128, 1]
        entropy = 0.5 * (torch.log(2.0 * np.pi * sigma_v ** 2) + 1.0).sum(dim=-1)

        return action_v, log_probs, entropy

    # https://github.com/pranz24/pytorch-soft-actor-critic/blob/398595e0d9dca98b7db78c7f2f939c969431871a/model.py#L64
    # def re_parameterization_trick_sample(self, obs):
    #     mu_v, sigma_v = self.actor_model.pi(obs)
    #     normal = Normal(loc=mu_v, scale=sigma_v)
    #     action_v = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
    #     action_v = torch.tanh(action_v)
    #
    #     log_prob = normal.log_prob(action_v)
    #     # Enforcing Action Bound
    #     epsilon = 1e-06
    #     log_prob = log_prob - torch.log(1.0 - action_v.pow(2) + epsilon)
    #     log_prob = log_prob.sum(dim=-1, keepdim=True)
    #     return action_v, log_prob


if __name__ == "__main__":
    a = DoubleQCriticModel((4,), 1, 4, config=config)