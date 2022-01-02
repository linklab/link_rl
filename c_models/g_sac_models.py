from typing import Tuple
import torch
from torch import nn
import numpy as np
from a_configuration.b_base.c_models.convolutional_models import ParameterConvolutionalModel
from a_configuration.b_base.c_models.linear_models import ParameterLinearModel
from a_configuration.b_base.c_models.recurrent_models import ParameterRecurrentModel
from c_models.a_models import Model
from c_models.c_policy_models import DiscreteActorModel, ContinuousActorModel
from torch.distributions import Normal, TanhTransform, TransformedDistribution
from e_main.parameter import parameter
from gym.spaces import Discrete, Box


class SacCriticModel(Model):
    def __init__(
            self, observation_shape: Tuple[int], n_out_actions: int, n_discrete_actions=None,
            device=torch.device("cpu"), parameter=None
    ):
        super(SacCriticModel, self).__init__(observation_shape, n_out_actions, n_discrete_actions, device, parameter)

        self.critic_params = []

        self.q1 = self.get_critic_models("q1")
        self.q2 = self.get_critic_models("q2")

        self.critic_params += list(self.q1.parameters())
        self.critic_params += list(self.q2.parameters())

    def get_critic_models(self, name):
        if isinstance(self.parameter.MODEL, ParameterLinearModel):
            input_n_features = self.observation_shape[0] + self.n_out_actions
            critic_layers = self.get_linear_layers(input_n_features=input_n_features)
        elif isinstance(self.parameter.MODEL, ParameterConvolutionalModel):
            input_n_channels = self.observation_shape[0] + self.n_out_actions
            critic_layers = self.get_conv_layers(input_n_channels=input_n_channels)
            conv_out_flat_size = self._get_conv_out(self.critic_conv_layers, self.observation_shape)
            critic_layers = nn.Sequential(
                critic_layers,
                self.get_linear_layers(input_n_features=conv_out_flat_size)
            )
        elif isinstance(self.parameter.MODEL, ParameterRecurrentModel):
            critic_layers = None
        else:
            raise ValueError()

        critic_layers.add_module(
            "critic_fc_last_{0}".format(name), nn.Linear(self.parameter.MODEL.NEURONS_PER_FULLY_CONNECTED_LAYER[-1], 1)
        )

        return critic_layers

    def q(self, obs, act):
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float32, device=self.device)

        x = torch.cat([obs, act], dim=-1)
        return self.q1(x), self.q2(x)


class DiscreteSacModel:
    def __init__(
            self, observation_shape: Tuple[int], n_out_actions: int, n_discrete_actions=None,
            device=torch.device("cpu"), parameter=None, is_target_model=False
    ):
        if is_target_model:
            self.actor_model = None
        else:
            self.actor_model = DiscreteActorModel(
                observation_shape=observation_shape, n_out_actions=n_out_actions, device=device, parameter=parameter
            ).to(device)

        self.critic_model = SacCriticModel(
            observation_shape=observation_shape, n_out_actions=n_out_actions, n_discrete_actions=n_discrete_actions,
            device=device, parameter=parameter
        ).to(device)


class ContinuousSacModel:
    def __init__(
            self, observation_shape: Tuple[int], n_out_actions: int, device=torch.device("cpu"),
            parameter=None, is_target_model=False
    ):
        if is_target_model:
            self.actor_model = None
        else:
            self.actor_model = ContinuousActorModel(
                observation_shape=observation_shape, n_out_actions=n_out_actions, device=device, parameter=parameter
            ).to(device)

        self.critic_model = SacCriticModel(
            observation_shape=observation_shape, n_out_actions=n_out_actions, n_discrete_actions=None,
            device=device, parameter=parameter
        ) .to(device)

    def re_parameterization_trick_sample(self, obs):
        mu_v, std_v = self.actor_model.pi(obs)

        dist = Normal(loc=mu_v, scale=std_v)
        dist = TransformedDistribution(base_distribution=dist, transforms=TanhTransform(cache_size=1))

        action_v = dist.rsample()  # for reparameterization trick (mean + std * N(0,1))
        log_probs = dist.log_prob(action_v).sum(dim=-1, keepdim=True)

        # action_v.shape: [128, 1]
        # log_prob.shape: [128, 1]
        return action_v, log_probs


if __name__ == "__main__":
    a = SacCriticModel((4,), 1, 4, device=torch.device("cpu"), parameter=parameter)