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

class SacCriticModel(Model):
    def __init__(
            self, observation_shape: Tuple[int], n_out_actions: int, n_discrete_actions=None,
            device=torch.device("cpu"), parameter=None
    ):
        super(SacCriticModel, self).__init__(observation_shape, n_out_actions, n_discrete_actions, device, parameter)
        self.n_discrete_actions = n_discrete_actions
        self.critic_params = []

        self.q1 = self.get_critic_models()
        self.q2 = self.get_critic_models()

    def get_critic_models(self):
        if isinstance(self.parameter.MODEL, ParameterLinearModel):
            input_n_features = self.observation_shape[0] + self.n_discrete_actions
            critic_layers = self.get_linear_layers(input_n_features=input_n_features)
        elif isinstance(self.parameter.MODEL, ParameterConvolutionalModel):
            input_n_channels = self.observation_shape[0] + self.n_discrete_actions
            critic_layers = self.get_conv_layers(input_n_channels=input_n_channels)
            conv_out_flat_size = self._get_conv_out(self.critic_conv_layers, self.observation_shape)
            critic_layers = nn.Sequential(
                critic_layers,
                self.get_linear_layers(input_n_features=conv_out_flat_size)
            )
        elif isinstance(self.parameter.MODEL, ParameterRecurrentModel):
            pass
        else:
            raise ValueError()
        critic_layers.add_module(
            "critic_fc_last", nn.Linear(self.parameter.MODEL.NEURONS_PER_FULLY_CONNECTED_LAYER[-1], 1)
        )
        self.critic_params += list(critic_layers.parameters())
        return critic_layers

    def v(self, obs, act):
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
        x = torch.cat([obs, act], dim=-1)
        return self.q1(x), self.q2(x)


class DiscreteSacModel(DiscreteActorModel, SacCriticModel):
    def __init__(
            self, observation_shape: Tuple[int], n_out_actions: int, n_discrete_actions=None,
            device=torch.device("cpu"), parameter=None
    ):
        super(DiscreteSacModel, self).__init__(observation_shape, n_out_actions, n_discrete_actions, device, parameter)


class ContinuousSacModel(ContinuousActorModel, SacCriticModel):
    def __init__(
            self, observation_shape: Tuple[int], n_out_actions: int, device=torch.device("cpu"), parameter=None
    ):
        super(ContinuousSacModel, self).__init__(observation_shape, n_out_actions, device, parameter)

    def pi(self, x):
        x = self.forward_actor(x)
        mu_v = self.mu(x)
        logstd_v = self.logstd(x)
        return mu_v, logstd_v

    def re_parameterization_trick_sample(self, state):
        mu_v, logstd_v, _ = self.base.forward_actor(state)
        dist = Normal(loc=mu_v, scale=torch.exp(logstd_v))
        transforms = [TanhTransform(cache_size=1)]
        dist = TransformedDistribution(dist, transforms)
        action_v = dist.rsample()  # for reparameterization trick (mean + std * N(0,1))

        log_probs = dist.log_prob(action_v).sum(dim=-1, keepdim=True)

        # action_v.shape: [128, 1]
        # log_prob.shape: [128, 1]
        return action_v, log_probs