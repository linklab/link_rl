from typing import Tuple
import numpy as np
import torch
from torch import nn

from a_configuration.b_base.c_models.convolutional_models import ParameterConvolutionalModel
from a_configuration.b_base.c_models.linear_models import ParameterLinearModel
from a_configuration.b_base.c_models.recurrent_models import ParameterRecurrentModel
from c_models.a_models import Model
from c_models.c_policy_models import DiscreteActorModel, ContinuousActorModel, PolicyModel


class ContinuousDdpgActorModel(PolicyModel):
    def __init__(
            self, observation_shape: Tuple[int], n_out_actions: int, parameter=None
    ):
        super(ContinuousDdpgActorModel, self).__init__(
            observation_shape=observation_shape, n_out_actions=n_out_actions, parameter=parameter
        )

        self.mu = nn.Sequential(
            nn.Linear(self.parameter.MODEL.NEURONS_PER_FULLY_CONNECTED_LAYER[-1], self.n_out_actions),
            nn.Tanh()
        )
        self.actor_params += list(self.mu.parameters())

    def pi(self, x):
        x = self.forward_actor(x)
        mu_v = self.mu(x)
        return mu_v


class DdpgCriticModel(Model):
    def __init__(
            self, observation_shape: Tuple[int], n_out_actions: int, n_discrete_actions=None, parameter=None
    ):
        super(DdpgCriticModel, self).__init__(observation_shape, n_out_actions, n_discrete_actions, parameter)
        #######################
        # CRITIC MODEL: BEGIN #
        #######################
        self.critic_params = []
        if isinstance(self.parameter.MODEL, ParameterLinearModel):
            input_n_features = self.observation_shape[0]
            self.critic_fc_layers = self.get_linear_layers(input_n_features=input_n_features + n_out_actions)
            self.critic_params += list(self.critic_fc_layers.parameters())
        elif isinstance(self.parameter.MODEL, ParameterConvolutionalModel):
            input_n_channels = self.observation_shape[0]
            self.critic_conv_layers = self.get_conv_layers(input_n_channels=input_n_channels)
            self.critic_params += list(self.critic_conv_layers.parameters())
            critic_conv_out_flat_size = self._get_conv_out(self.conv_layers, observation_shape)
            self.critic_fc_layers = self.get_linear_layers(input_n_features=critic_conv_out_flat_size + n_out_actions)
            self.critic_params += list(self.critic_fc_layers.parameters())
        elif isinstance(self.parameter.MODEL, ParameterRecurrentModel):
            pass
        else:
            raise ValueError()

        self.critic_fc_last = nn.Linear(self.parameter.MODEL.NEURONS_PER_FULLY_CONNECTED_LAYER[-1], 1)
        self.critic_params += list(self.critic_fc_last.parameters())
        #####################
        # CRITIC MODEL: END #
        #####################

    def forward_critic(self, x, a):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32, device=self.parameter.DEVICE)

        if isinstance(a, np.ndarray):
            a = torch.tensor(a, dtype=torch.float32, device=self.parameter.DEVICE)

        if isinstance(self.parameter.MODEL, ParameterLinearModel):
            x = self.critic_fc_layers(torch.cat([x, a], dim=-1))
        elif isinstance(self.parameter.MODEL, ParameterConvolutionalModel):
            conv_out = self.critic_conv_layers(x)
            conv_out = torch.flatten(conv_out, start_dim=1)
            x = self.critic_fc_layers(torch.cat([conv_out, a], dim=-1))
        else:
            raise ValueError()
        return x

    def q(self, x, a):
        x = self.forward_critic(x, a)
        q_value = self.critic_fc_last(x)
        return q_value


class DiscreteDdpgModel:
    def __init__(
            self, observation_shape: Tuple[int], n_out_actions: int, n_discrete_actions=None, parameter=None
    ):
        self.parameter = parameter

        self.actor_model = DiscreteActorModel(
            observation_shape=observation_shape, n_out_actions=n_out_actions, parameter=self.parameter
        ).to(self.parameter.DEVICE)

        self.critic_model = DdpgCriticModel(
            observation_shape=observation_shape, n_out_actions=n_out_actions, n_discrete_actions=n_discrete_actions,
            parameter=self.parameter
        ).to(self.parameter.DEVICE)


class ContinuousDdpgModel:
    def __init__(
            self, observation_shape: Tuple[int], n_out_actions: int, parameter=None
    ):
        self.parameter = parameter

        self.actor_model = ContinuousDdpgActorModel(
            observation_shape=observation_shape, n_out_actions=n_out_actions, parameter=self.parameter
        ).to(self.parameter.DEVICE)

        self.critic_model = DdpgCriticModel(
            observation_shape=observation_shape, n_out_actions=n_out_actions, n_discrete_actions=None,
            parameter=self.parameter
        ).to(self.parameter.DEVICE)
