from typing import Tuple
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from a_configuration.b_base.c_models.convolutional_models import ParameterConvolutionalModel
from a_configuration.b_base.c_models.linear_models import ParameterLinearModel
from c_models.a_models import Model
from g_utils.types import ModelType


class ActorCritic(Model):
    def __init__(
            self, observation_shape: Tuple[int], n_out_actions: int, device=torch.device("cpu"), parameter=None
    ):
        super(ActorCritic, self).__init__(observation_shape, n_out_actions, device, parameter)

        if isinstance(self.parameter.MODEL, ParameterLinearModel):
            input_n_features = self.observation_shape[0]
            self.actor_fc_layers = self.get_linear_layers(input_n_features=input_n_features)
            self.critic_fc_layers = self.get_linear_layers(input_n_features=input_n_features)

            self.actor_params = list(self.actor_fc_layers.parameters())
            self.critic_params = list(self.critic_fc_layers.parameters())
        elif isinstance(self.parameter.MODEL, ParameterConvolutionalModel):
            input_n_channels = self.observation_shape[0]
            self.actor_conv_layers = self.get_conv_layers(input_n_channels=input_n_channels)
            conv_out_flat_size = self._get_conv_out(observation_shape)
            self.actor_fc_layers = self.get_linear_layers(input_n_features=conv_out_flat_size)

            self.critic_conv_layers = self.get_conv_layers(input_n_channels=input_n_channels)
            conv_out_flat_size = self._get_conv_out(observation_shape)
            self.critic_fc_layers = self.get_linear_layers(input_n_features=conv_out_flat_size)

            self.actor_params = list(self.actor_conv_layers.parameters()) + list(self.actor_fc_layers.parameters())
            self.critic_params = list(self.critic_conv_layers.parameters()) + list(self.critic_fc_layers.parameters())
        else:
            raise ValueError()

        self.critic_fc_v = nn.Linear(self.parameter.MODEL.NEURONS_PER_FULLY_CONNECTED_LAYER[-1], 1)
        self.critic_params += list(self.critic_fc_v.parameters())

    def forward_actor(self, x):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32, device=self.device)

        if isinstance(self.parameter.MODEL, ParameterLinearModel):
            x = self.actor_fc_layers(x)
        elif isinstance(self.parameter.MODEL, ParameterConvolutionalModel):
            conv_out = self.actor_conv_layers(x)
            conv_out = torch.flatten(conv_out, start_dim=1)
            x = self.actor_fc_layers(conv_out)
        else:
            raise ValueError()
        return x

    def forward_critic(self, x):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32, device=self.device)

        if isinstance(self.parameter.MODEL, ParameterLinearModel):
            x = self.critic_fc_layers(x)
        elif isinstance(self.parameter.MODEL, ParameterConvolutionalModel):
            conv_out = self.critic_conv_layers(x)
            conv_out = torch.flatten(conv_out, start_dim=1)
            x = self.critic_fc_layers(conv_out)
        else:
            raise ValueError()
        return x

    def v(self, x):
        x = self.forward_critic(x)
        v = self.critic_fc_v(x)
        return v


class DiscreteActorCritic(ActorCritic):
    def __init__(
            self, observation_shape: Tuple[int], n_out_actions: int, device=torch.device("cpu"), parameter=None
    ):
        super(DiscreteActorCritic, self).__init__(observation_shape, n_out_actions, device, parameter)

        self.actor_fc_pi = nn.Linear(self.parameter.MODEL.NEURONS_PER_FULLY_CONNECTED_LAYER[-1], self.n_out_actions)
        self.actor_params += list(self.actor_fc_pi.parameters())

    def pi(self, x):
        x = self.forward_actor(x)
        x = self.actor_fc_pi(x)
        prob = F.softmax(x, dim=-1)
        return prob


class ContinuousActorCritic(ActorCritic):
    def __init__(
            self, observation_shape: Tuple[int], n_out_actions: int, device=torch.device("cpu"), parameter=None
    ):
        super(ContinuousActorCritic, self).__init__(observation_shape, n_out_actions, device, parameter)

        self.mu = nn.Sequential(
            nn.Linear(self.parameter.MODEL.NEURONS_PER_FULLY_CONNECTED_LAYER[-1], self.n_out_actions),
            nn.Tanh()
        )

        logstds_param = nn.Parameter(torch.full((self.n_out_actions,), 0.1))
        self.register_parameter("logstds", logstds_param)
        self.actor_params += list(self.mu.parameters())
        self.actor_params.append(self.logstds)

        # self.logstd = nn.Sequential(
        #     nn.Linear(self.parameter.MODEL.NEURONS_PER_FULLY_CONNECTED_LAYER[-1], self.n_out_actions),
        #     nn.Softplus()
        # )
        # self.actor_params += list(self.mu.parameters())
        # self.actor_params += list(self.logstd.parameters())

    def pi(self, x):
        x = self.forward_actor(x)
        mu_v = self.mu(x)
        std_v = F.softplus(self.logstds.exp())
        #std_v = self.logstd(x).exp()
        return mu_v, std_v
