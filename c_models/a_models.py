import torch
from torch import nn
from typing import Tuple
from collections import OrderedDict
import numpy as np


class Model(nn.Module):
    def __init__(
            self, observation_shape: Tuple[int], n_out_actions: int, device=torch.device("cpu"), parameter=None
    ):
        super(Model, self).__init__()
        self.observation_shape = observation_shape
        self.n_out_actions = n_out_actions
        self.device = device
        self.parameter = parameter

    def get_linear_layers(self, input_n_features):
        assert self.parameter.NEURONS_PER_FULLY_CONNECTED_LAYER

        fc_layers_dict = OrderedDict()
        fc_layers_dict["fc_0"] = nn.Linear(input_n_features, self.parameter.NEURONS_PER_FULLY_CONNECTED_LAYER[0])
        fc_layers_dict["fc_0_activation"] = nn.LeakyReLU()

        for idx in range(1, len(self.parameter.NEURONS_PER_FULLY_CONNECTED_LAYER) - 1):
            fc_layers_dict["fc_{0}".format(idx)] = nn.Linear(
                self.parameter.NEURONS_PER_FULLY_CONNECTED_LAYER[idx],
                self.parameter.NEURONS_PER_FULLY_CONNECTED_LAYER[idx + 1]
            )
            fc_layers_dict["fc_{0}_activation".format(idx)] = nn.LeakyReLU()

        fc_layers = nn.Sequential(fc_layers_dict)
        return fc_layers

    def get_conv_layers(self, input_n_channels):
        assert self.parameter.OUT_CHANNELS_PER_LAYER
        assert self.parameter.KERNEL_SIZE_PER_LAYER
        assert self.parameter.STRIDE_PER_LAYER
        assert self.parameter.NEURONS_PER_FULLY_CONNECTED_LAYER

        conv_layers_dict = OrderedDict()
        conv_layers_dict["conv_0"] = nn.Conv2d(
            in_channels=input_n_channels,
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

        conv_layers = nn.Sequential(conv_layers_dict)
        return conv_layers

    def _get_conv_out(self, shape):
        cont_out = self.conv_layers(torch.zeros(1, *shape))
        return int(np.prod(cont_out.size()))