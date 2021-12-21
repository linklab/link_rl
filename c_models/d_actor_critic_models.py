from typing import Tuple
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from c_models.a_models import Model
from g_utils.types import ModelType


class ActorCritic(Model):
    def __init__(
            self, observation_shape: Tuple[int], n_out_actions: int, device=torch.device("cpu"), parameter=None
    ):
        super(ActorCritic, self).__init__(observation_shape, n_out_actions, device, parameter)

        if self.parameter.MODEL_TYPE == ModelType.LINEAR:
            input_n_features = self.observation_shape[0]
            self.fc_layers = self.get_linear_layers(input_n_features=input_n_features)
        elif self.parameter.MODEL_TYPE == ModelType.CONVOLUTIONAL:
            input_n_channels = self.observation_shape[0]
            self.conv_layers = self.get_conv_layers(input_n_channels=input_n_channels)
            conv_out_flat_size = self._get_conv_out(observation_shape)
            self.fc_layers = self.get_linear_layers(input_n_features=conv_out_flat_size)
        else:
            raise ValueError()

        self.fc_pi = nn.Linear(self.parameter.NEURONS_PER_FULLY_CONNECTED_LAYER[-1], self.n_out_actions)
        self.fc_v = nn.Linear(self.parameter.NEURONS_PER_FULLY_CONNECTED_LAYER[-1], 1)

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32, device=self.device)

        if self.parameter.MODEL_TYPE == ModelType.LINEAR:
            x = self.fc_layers(x)
        elif self.parameter.MODEL_TYPE == ModelType.CONVOLUTIONAL:
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
            nn.Linear(self.parameter.NEURONS_PER_FULLY_CONNECTED_LAYER[-1], self.n_out_actions),
            nn.Tanh()
        )

        self.logstd = nn.Sequential(
            nn.Linear(self.parameter.NEURONS_PER_FULLY_CONNECTED_LAYER[-1], self.n_out_actions),
            nn.Softplus()
        )

    def pi(self, x):
        x = self.forward(x)
        mu_v = self.mu(x)
        logstd_v = self.logstd(x)
        return mu_v, logstd_v

    def v(self, x):
        x = self.forward(x)
        v = self.fc_v(x)
        return v