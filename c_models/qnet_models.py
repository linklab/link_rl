import random
from collections import OrderedDict
from typing import Tuple
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Categorical

from g_utils.types import AgentMode, ModelType


class QNet(nn.Module):
    def __init__(
            self, observation_shape: Tuple[int], n_out_actions: int, device=torch.device("cpu"), parameter=None
    ):
        super(QNet, self).__init__()
        self.observation_shape = observation_shape
        self.n_out_actions = n_out_actions
        self.device = device
        self.parameter = parameter

        if self.parameter.MODEL_TYPE == ModelType.LINEAR:
            assert self.parameter.NEURONS_PER_LAYER

            n_features = self.observation_shape[0]

            fc_layers_dict = OrderedDict()
            fc_layers_dict["fc_0"] = nn.Linear(n_features, self.parameter.NEURONS_PER_LAYER[0])
            fc_layers_dict["fc_0_activation"] = nn.LeakyReLU()

            for idx in range(1, len(self.parameter.NEURONS_PER_LAYER) - 1):
                fc_layers_dict["fc_{0}".format(idx)] = nn.Linear(
                    self.parameter.NEURONS_PER_LAYER[idx], self.parameter.NEURONS_PER_LAYER[idx + 1]
                )
                fc_layers_dict["fc_{0}_activation".format(idx)] = nn.LeakyReLU()

            self.fc_layers = nn.Sequential(fc_layers_dict)
            self.fc_last = nn.Linear(self.parameter.NEURONS_PER_LAYER[-1], self.n_out_actions)

        elif self.parameter.MODEL_TYPE == ModelType.CONVOLUTIONAL:
            assert self.parameter.OUT_CHANNELS_PER_LAYER
            assert self.parameter.KERNEL_SIZE_PER_LAYER
            assert self.parameter.STRIDE_PER_LAYER
            assert self.parameter.NEURONS_PER_FULLY_CONNECTED_LAYER

            input_channel = self.observation_shape[0]

            conv_layers_dict = OrderedDict()
            conv_layers_dict["conv_0"] = nn.Conv2d(
                in_channels=input_channel,
                out_channels=self.parameter.OUT_CHANNELS_PER_LAYER[0],
                kernel_size=self.parameter.KERNEL_SIZE_PER_LAYER[0],
                stride=self.parameter.STRIDE_PER_LAYER[0]
            )
            conv_layers_dict["conv_0_activation"] = nn.LeakyReLU()

            for idx in range(1, len(self.parameter.OUT_CHANNELS_PER_LAYER)):
                conv_layers_dict["conv_{0}".format(idx)] = nn.Conv2d(
                    in_channels=self.parameter.OUT_CHANNELS_PER_LAYER[idx-1],
                    out_channels=self.parameter.OUT_CHANNELS_PER_LAYER[idx],
                    kernel_size=self.parameter.KERNEL_SIZE_PER_LAYER[idx],
                    stride=self.parameter.STRIDE_PER_LAYER[idx]
                )
                conv_layers_dict["conv_{0}_activation".format(idx)] = nn.LeakyReLU()

            self.conv_layers = nn.Sequential(conv_layers_dict)
            conv_out_flat_size = self._get_conv_out(observation_shape)

            fc_layers_dict = OrderedDict()
            fc_layers_dict["fc_0"] = nn.Linear(
                conv_out_flat_size, self.parameter.NEURONS_PER_FULLY_CONNECTED_LAYER[0]
            )
            fc_layers_dict["fc_0_activation"] = nn.LeakyReLU()

            if len(self.parameter.NEURONS_PER_FULLY_CONNECTED_LAYER) >= 2:
                for idx in range(1, len(self.parameter.NEURONS_PER_FULLY_CONNECTED_LAYER) - 1):
                    fc_layers_dict["fc_{0}".format(idx)] = nn.Linear(
                        self.parameter.NEURONS_PER_FULLY_CONNECTED_LAYER[idx],
                        self.parameter.NEURONS_PER_FULLY_CONNECTED_LAYER[idx + 1]
                    )
                    fc_layers_dict["fc_{0}_activation".format(idx)] = nn.LeakyReLU()

            self.fc_layers = nn.Sequential(fc_layers_dict)
            self.fc_last = nn.Linear(
                self.parameter.NEURONS_PER_FULLY_CONNECTED_LAYER[-1], n_out_actions
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
