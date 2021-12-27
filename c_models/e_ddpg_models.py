from typing import Tuple
import numpy as np
import torch
from torch import nn

from a_configuration.b_base.c_models.convolutional_models import ParameterConvolutionalModel
from a_configuration.b_base.c_models.linear_models import ParameterLinearModel
from a_configuration.b_base.c_models.recurrent_models import ParameterRecurrentModel
from c_models.a_models import Model
from c_models.c_policy_models import DiscreteActorModel, ContinuousActorModel


class DdpgCriticModel(Model):
    def __init__(
            self, observation_shape: Tuple[int], n_out_actions: int, device=torch.device("cpu"), parameter=None
    ):
        super(DdpgCriticModel, self).__init__(observation_shape, n_out_actions, device, parameter)

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
        self.critic_params += list(self.critic_fc_v.parameters())
        #####################
        # CRITIC MODEL: END #
        #####################

    def forward_critic(self, x, a):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32, device=self.device)

        if isinstance(a, np.ndarray):
            a = torch.tensor(a, dtype=torch.float32, device=self.device)

        if isinstance(self.parameter.MODEL, ParameterLinearModel):
            x = self.critic_fc_layers(torch.cat([x, a], dim=-1))
        elif isinstance(self.parameter.MODEL, ParameterConvolutionalModel):
            conv_out = self.critic_conv_layers(x)
            conv_out = torch.flatten(conv_out, start_dim=1)
            x = self.critic_fc_layers(torch.cat([conv_out, a], dim=-1))
        else:
            raise ValueError()
        return x

    def v(self, x, a):
        x = self.forward_critic(x, a)
        v = self.critic_fc_last(x)
        return v


class DiscreteDdpgModel(DiscreteActorModel, DdpgCriticModel):
    def __init__(
            self, observation_shape: Tuple[int], n_out_actions: int, device=torch.device("cpu"), parameter=None
    ):
        super(DiscreteDdpgModel, self).__init__(observation_shape, n_out_actions, device, parameter)


class ContinuousDdpgModel(ContinuousActorModel, DdpgCriticModel):
    def __init__(
            self, observation_shape: Tuple[int], n_out_actions: int, device=torch.device("cpu"), parameter=None
    ):
        super(ContinuousDdpgModel, self).__init__(observation_shape, n_out_actions, device, parameter)
