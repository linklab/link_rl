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
            self.fc_layers = self.get_linear_layers(input_n_features=input_n_features)
        elif isinstance(self.parameter.MODEL, ParameterConvolutionalModel):
            input_n_channels = self.observation_shape[0]
            self.conv_layers = self.get_conv_layers(input_n_channels=input_n_channels)
            conv_out_flat_size = self._get_conv_out(observation_shape)
            self.fc_layers = self.get_linear_layers(input_n_features=conv_out_flat_size)
        else:
            raise ValueError()

        self.fc_pi = nn.Linear(self.parameter.MODEL.NEURONS_PER_FULLY_CONNECTED_LAYER[-1], self.n_out_actions)
        self.fc_v = nn.Linear(self.parameter.MODEL.NEURONS_PER_FULLY_CONNECTED_LAYER[-1], 1)

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32, device=self.device)

        if isinstance(self.parameter.MODEL, ParameterLinearModel):
            x = self.fc_layers(x)
        elif isinstance(self.parameter.MODEL, ParameterConvolutionalModel):
            conv_out = self.conv_layers(x)
            conv_out = torch.flatten(conv_out, start_dim=1)
            x = self.fc_layers(conv_out)
        else:
            raise ValueError()

        return x

    def pi(self, x):
        x = self.forward(x)
        x = self.fc_pi(x)
        prob = F.softmax(x, dim=-1)
        return prob

    def v(self, x):
        x = self.forward(x)
        v = self.fc_v(x)
        return v


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

        # self.logstd = nn.Sequential(
        #     nn.Linear(self.parameter.MODEL.NEURONS_PER_FULLY_CONNECTED_LAYER[-1], self.n_out_actions),
        #     nn.Softplus()
        # )

    def pi(self, x):
        x = self.forward(x)
        mu_v = self.mu(x)
        std_v = F.softplus(self.logstds).exp()
        # std_v = self.logstd(x).exp()
        return mu_v, std_v
