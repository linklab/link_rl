import torch
from torch import nn
from typing import Tuple
from collections import OrderedDict
import numpy as np

from a_configuration.b_base.c_models.convolutional_models import ParameterConvolutionalModel
from a_configuration.b_base.c_models.linear_models import ParameterLinearModel
from a_configuration.b_base.c_models.recurrent_convolutional_models import ParameterRecurrentConvolutionalModel
from a_configuration.b_base.c_models.recurrent_linear_models import ParameterRecurrentLinearModel


class Model(nn.Module):
    def __init__(
            self, observation_shape: Tuple[int], n_out_actions: int, n_discrete_actions=None, parameter=None
    ):
        super(Model, self).__init__()
        self.observation_shape = observation_shape
        self.n_out_actions = n_out_actions
        self.n_discrete_actions = n_discrete_actions
        self.parameter = parameter

        self.is_recurrent_model = any([
            isinstance(self.parameter.MODEL_PARAMETER, ParameterRecurrentLinearModel),
            isinstance(self.parameter.MODEL_PARAMETER, ParameterRecurrentConvolutionalModel)
        ])
        if self.is_recurrent_model:
            self.recurrent_hidden = None
            self.init_recurrent_hidden()

    def init_recurrent_hidden(self):
        if self.is_recurrent_model:
            self.recurrent_hidden = torch.zeros(
                self.parameter.MODEL_PARAMETER.NUM_LAYERS,
                1,  # batch_size
                self.parameter.MODEL_PARAMETER.HIDDEN_SIZE,
                dtype=torch.float32,
                device=self.parameter.DEVICE
            )

    def get_linear_layers(self, input_n_features):
        assert self.parameter.MODEL_PARAMETER.NEURONS_PER_FULLY_CONNECTED_LAYER

        fc_layers_dict = OrderedDict()
        fc_layers_dict["fc_0"] = nn.Linear(
            input_n_features,
            self.parameter.MODEL_PARAMETER.NEURONS_PER_FULLY_CONNECTED_LAYER[0]
        )
        if self.parameter.USE_LAYER_NORM:
            self.get_layer_normalization(fc_layers_dict, 0)

        fc_layers_dict["fc_0_activation"] = self.parameter.LAYER_ACTIVATION()

        for idx in range(1, len(self.parameter.MODEL_PARAMETER.NEURONS_PER_FULLY_CONNECTED_LAYER)):
            fc_layers_dict["fc_{0}".format(idx)] = nn.Linear(
                self.parameter.MODEL_PARAMETER.NEURONS_PER_FULLY_CONNECTED_LAYER[idx - 1],
                self.parameter.MODEL_PARAMETER.NEURONS_PER_FULLY_CONNECTED_LAYER[idx]
            )
            if self.parameter.USE_LAYER_NORM:
                self.get_layer_normalization(fc_layers_dict, idx)

            fc_layers_dict["fc_{0}_activation".format(idx)] = self.parameter.LAYER_ACTIVATION()

        fc_layers = nn.Sequential(fc_layers_dict)

        return fc_layers

    def get_conv_layers(self, input_n_channels):
        assert self.parameter.MODEL_PARAMETER.OUT_CHANNELS_PER_LAYER
        assert self.parameter.MODEL_PARAMETER.KERNEL_SIZE_PER_LAYER
        assert self.parameter.MODEL_PARAMETER.STRIDE_PER_LAYER
        assert self.parameter.MODEL_PARAMETER.NEURONS_PER_FULLY_CONNECTED_LAYER

        conv_layers_dict = OrderedDict()
        conv_layers_dict["conv_0"] = nn.Conv2d(
            in_channels=input_n_channels,
            out_channels=self.parameter.MODEL_PARAMETER.OUT_CHANNELS_PER_LAYER[0],
            kernel_size=self.parameter.MODEL_PARAMETER.KERNEL_SIZE_PER_LAYER[0],
            stride=self.parameter.MODEL_PARAMETER.STRIDE_PER_LAYER[0]
        )
        conv_layers_dict["conv_0_activation"] = self.parameter.LAYER_ACTIVATION()

        for idx in range(1, len(self.parameter.MODEL_PARAMETER.OUT_CHANNELS_PER_LAYER)):
            conv_layers_dict["conv_{0}".format(idx)] = nn.Conv2d(
                in_channels=self.parameter.MODEL_PARAMETER.OUT_CHANNELS_PER_LAYER[idx - 1],
                out_channels=self.parameter.MODEL_PARAMETER.OUT_CHANNELS_PER_LAYER[idx],
                kernel_size=self.parameter.MODEL_PARAMETER.KERNEL_SIZE_PER_LAYER[idx],
                stride=self.parameter.MODEL_PARAMETER.STRIDE_PER_LAYER[idx]
            )
            conv_layers_dict["conv_{0}_activation".format(idx)] = self.parameter.LAYER_ACTIVATION()

        conv_layers = nn.Sequential(conv_layers_dict)
        return conv_layers

    def _get_conv_out(self, conv_layers, shape):
        cont_out = conv_layers(torch.zeros(1, *shape))
        return int(np.prod(cont_out.size()))

    def get_layer_normalization(self, layer_dict, layer_idx):
        if isinstance(self.parameter.MODEL_PARAMETER, ParameterLinearModel):
            fc_layers_dict = layer_dict
            fc_layers_dict["fc_{0}_norm".format(layer_idx)] = nn.LayerNorm(
                self.parameter.MODEL_PARAMETER.NEURONS_PER_FULLY_CONNECTED_LAYER[layer_idx]
            )
        elif isinstance(self.parameter.MODEL_PARAMETER, ParameterConvolutionalModel):
            pass
        elif isinstance(self.parameter.MODEL_PARAMETER, ParameterRecurrentLinearModel):
            pass

    def get_recurrent_layers(self, input_n_features):
        assert self.parameter.MODEL_PARAMETER.HIDDEN_SIZE
        assert self.parameter.MODEL_PARAMETER.NUM_LAYERS
        assert self.parameter.MODEL_PARAMETER.NEURONS_PER_FULLY_CONNECTED_LAYER

        rnn_layer = nn.GRU(
            input_size=input_n_features,
            hidden_size=self.parameter.MODEL_PARAMETER.HIDDEN_SIZE,
            num_layers=self.parameter.MODEL_PARAMETER.NUM_LAYERS,
            batch_first=True,
            bidirectional=False
        )

        return rnn_layer

    # def _get_recurrent_out(self, recurrent_layers, input_n_features, seq_len=1):
    #     """
    #     The inputs of the RNN:
    #         rnn_in (batch_size, sequence_length, HIDDEN_SIZE): the input of the sequences
    #         h_0 (NUM_LAYERS, batch_size, HIDDEN_SIZE): the input of the layers
    #     The outputs of the RNN:
    #         rnn_out (batch_size, sequence_length, HIDDEN_SIZE): the output of the sequences
    #         h_n (NUM_LAYERS, batch_size, HIDDEN_SIZE): the output of the layers
    #             Defaults to zeros if not provided(or provided None).
    #     """
    #     rnn_in = torch.zeros(1, seq_len, input_n_features)
    #     rnn_out, h_n = recurrent_layers(rnn_in)
    #     return int(np.prod(rnn_out.size())), int(np.prod(h_n.size()))

