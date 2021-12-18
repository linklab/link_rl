from typing import Tuple
import numpy as np
import torch
from torch import nn

from c_models.a_models import Model
from g_utils.types import ModelType


class QNet(Model):
    def __init__(
            self, observation_shape: Tuple[int], n_out_actions: int, device=torch.device("cpu"), parameter=None
    ):
        super(QNet, self).__init__(observation_shape, n_out_actions, device, parameter)

        if self.parameter.MODEL_TYPE == ModelType.LINEAR:
            input_n_features = self.observation_shape[0]
            self.fc_layers = self.get_linear_layers(input_n_features=input_n_features)
            self.fc_last = nn.Linear(
                self.parameter.NEURONS_PER_FULLY_CONNECTED_LAYER[-1], self.n_out_actions
            )
        elif self.parameter.MODEL_TYPE == ModelType.CONVOLUTIONAL:
            input_n_channels = self.observation_shape[0]
            self.conv_layers = self.get_conv_layers(input_n_channels=input_n_channels)
            conv_out_flat_size = self._get_conv_out(observation_shape)
            self.fc_layers = self.get_linear_layers(input_n_features=conv_out_flat_size)
            self.fc_last = nn.Linear(
                self.parameter.NEURONS_PER_FULLY_CONNECTED_LAYER[-1], self.n_out_actions
            )

        self.version = 0

    def _get_conv_out(self, shape):
        cont_out = self.conv_layers(torch.zeros(1, *shape))
        return int(np.prod(cont_out.size()))

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32, device=self.device)

        if self.parameter.MODEL_TYPE == ModelType.LINEAR:
            x = self.fc_layers(x)
            x = self.fc_last(x)
        elif self.parameter.MODEL_TYPE == ModelType.CONVOLUTIONAL:
            conv_out = self.conv_layers(x)
            conv_out = torch.flatten(conv_out, start_dim=1)
            x = self.fc_layers(conv_out)
        else:
            raise ValueError()

        x = self.fc_last(x)
        return x
