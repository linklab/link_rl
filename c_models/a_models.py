from abc import abstractmethod

import torch
from torch import nn
from typing import Tuple
from collections import OrderedDict
import numpy as np

from a_configuration.a_base_config.c_models.convolutional_models import ConfigConvolutionalModel
from a_configuration.a_base_config.c_models.linear_models import ConfigLinearModel
from a_configuration.a_base_config.c_models.recurrent_convolutional_models import ConfigRecurrentConvolutionalModel
from a_configuration.a_base_config.c_models.recurrent_linear_models import ConfigRecurrentLinearModel


class Model(nn.Module):
    def __init__(
        self,
        observation_shape: Tuple[int],
        n_out_actions: int,
        n_discrete_actions=None,
        config=None
    ):
        super(Model, self).__init__()
        self.observation_shape = observation_shape
        self.n_out_actions = n_out_actions
        self.n_discrete_actions = n_discrete_actions
        self.config = config

        self.is_recurrent_model = any([
            isinstance(self.config.MODEL_PARAMETER, ConfigRecurrentLinearModel),
            isinstance(self.config.MODEL_PARAMETER, ConfigRecurrentConvolutionalModel)
        ])
        if self.is_recurrent_model:
            self.recurrent_hidden = None
            self.init_recurrent_hidden()

    def init_recurrent_hidden(self):
        if self.is_recurrent_model:
            self.recurrent_hidden = torch.zeros(
                self.config.MODEL_PARAMETER.NUM_LAYERS,
                1,  # batch_size
                self.config.MODEL_PARAMETER.HIDDEN_SIZE,
                dtype=torch.float32,
                device=self.config.DEVICE
            )

    def get_linear_layers(self, input_n_features):
        assert self.config.MODEL_PARAMETER.NEURONS_PER_FULLY_CONNECTED_LAYER

        linear_layers_dict = OrderedDict()
        linear_layers_dict["linear_0"] = nn.Linear(
            input_n_features,
            self.config.MODEL_PARAMETER.NEURONS_PER_FULLY_CONNECTED_LAYER[0]
        )
        if self.config.USE_LAYER_NORM:
            self.get_layer_normalization(linear_layers_dict, 0)

        linear_layers_dict["linear_0_activation"] = self.config.LAYER_ACTIVATION()

        for idx in range(1, len(self.config.MODEL_PARAMETER.NEURONS_PER_FULLY_CONNECTED_LAYER)):
            linear_layers_dict["linear_{0}".format(idx)] = nn.Linear(
                self.config.MODEL_PARAMETER.NEURONS_PER_FULLY_CONNECTED_LAYER[idx - 1],
                self.config.MODEL_PARAMETER.NEURONS_PER_FULLY_CONNECTED_LAYER[idx]
            )
            if self.config.USE_LAYER_NORM:
                self.get_layer_normalization(linear_layers_dict, idx)

            linear_layers_dict["linear_{0}_activation".format(idx)] = self.config.LAYER_ACTIVATION()

        linear_layers = nn.Sequential(linear_layers_dict)

        return linear_layers

    def get_representation_layers(self, input_n_features):
        assert self.config.MODEL_PARAMETER.NEURONS_PER_REPRESENTATION_LAYER

        representation_layers_dict = OrderedDict()
        representation_layers_dict["representation_0"] = nn.Linear(
            input_n_features,
            self.config.MODEL_PARAMETER.NEURONS_PER_REPRESENTATION_LAYER[0]
        )
        if self.config.USE_LAYER_NORM:
            self.get_layer_normalization(representation_layers_dict, 0)

        representation_layers_dict["representation_0_activation"] = self.config.LAYER_ACTIVATION()

        for idx in range(1, len(self.config.MODEL_PARAMETER.NEURONS_PER_REPRESENTATION_LAYER)):
            representation_layers_dict["representation_{0}".format(idx)] = nn.Linear(
                self.config.MODEL_PARAMETER.NEURONS_PER_REPRESENTATION_LAYER[idx - 1],
                self.config.MODEL_PARAMETER.NEURONS_PER_REPRESENTATION_LAYER[idx]
            )
            if self.config.USE_LAYER_NORM:
                self.get_layer_normalization(representation_layers_dict, idx)

            representation_layers_dict["representation_{0}_activation".format(idx)] = self.config.LAYER_ACTIVATION()

        representation_layers = nn.Sequential(representation_layers_dict)

        return representation_layers

    def get_convolutional_layers(self, input_n_channels):
        assert self.config.MODEL_PARAMETER.OUT_CHANNELS_PER_LAYER
        assert self.config.MODEL_PARAMETER.KERNEL_SIZE_PER_LAYER
        assert self.config.MODEL_PARAMETER.STRIDE_PER_LAYER
        assert self.config.MODEL_PARAMETER.NEURONS_PER_FULLY_CONNECTED_LAYER

        convolutional_layers_dict = OrderedDict()
        convolutional_layers_dict["conv_0"] = nn.Conv2d(
            in_channels=input_n_channels,
            out_channels=self.config.MODEL_PARAMETER.OUT_CHANNELS_PER_LAYER[0],
            kernel_size=self.config.MODEL_PARAMETER.KERNEL_SIZE_PER_LAYER[0],
            stride=self.config.MODEL_PARAMETER.STRIDE_PER_LAYER[0]
        )
        convolutional_layers_dict["conv_0_activation"] = self.config.LAYER_ACTIVATION()

        for idx in range(1, len(self.config.MODEL_PARAMETER.OUT_CHANNELS_PER_LAYER)):
            convolutional_layers_dict["conv_{0}".format(idx)] = nn.Conv2d(
                in_channels=self.config.MODEL_PARAMETER.OUT_CHANNELS_PER_LAYER[idx - 1],
                out_channels=self.config.MODEL_PARAMETER.OUT_CHANNELS_PER_LAYER[idx],
                kernel_size=self.config.MODEL_PARAMETER.KERNEL_SIZE_PER_LAYER[idx],
                stride=self.config.MODEL_PARAMETER.STRIDE_PER_LAYER[idx]
            )
            convolutional_layers_dict["conv_{0}_activation".format(idx)] = self.config.LAYER_ACTIVATION()

        convolutional_layers = nn.Sequential(convolutional_layers_dict)
        return convolutional_layers

    def _get_conv_out(self, conv_layers, shape):
        conv_out = conv_layers(torch.zeros(1, *shape))
        return int(np.prod(conv_out.size()))

    def get_layer_normalization(self, layer_dict, layer_idx):
        if isinstance(self.config.MODEL_PARAMETER, ConfigLinearModel):
            linear_layers_dict = layer_dict
            linear_layers_dict["linear_{0}_norm".format(layer_idx)] = nn.LayerNorm(
                self.config.MODEL_PARAMETER.NEURONS_PER_FULLY_CONNECTED_LAYER[layer_idx]
            )
        elif isinstance(self.config.MODEL_PARAMETER, ConfigConvolutionalModel):
            pass
        elif isinstance(self.config.MODEL_PARAMETER, ConfigRecurrentLinearModel):
            pass

    def get_recurrent_layers(self, input_n_features):
        assert self.config.MODEL_PARAMETER.HIDDEN_SIZE
        assert self.config.MODEL_PARAMETER.NUM_LAYERS
        assert self.config.MODEL_PARAMETER.NEURONS_PER_FULLY_CONNECTED_LAYER

        rnn_layer = nn.GRU(
            input_size=input_n_features,
            hidden_size=self.config.MODEL_PARAMETER.HIDDEN_SIZE,
            num_layers=self.config.MODEL_PARAMETER.NUM_LAYERS,
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

    def make_linear_model(self, observation_shape, n_out_actions=None):
        input_n_features = observation_shape[0]
        self.representation_layers = self.get_representation_layers(input_n_features=input_n_features)

        if n_out_actions is None:
            input_n_features = self.config.MODEL_PARAMETER.NEURONS_PER_REPRESENTATION_LAYER[-1]
        else:
            input_n_features = self.config.MODEL_PARAMETER.NEURONS_PER_REPRESENTATION_LAYER[-1] + n_out_actions
        self.linear_layers = self.get_linear_layers(input_n_features=input_n_features)

    def make_convolutional_model(self, observation_shape, n_out_actions=None):
        input_n_channels = observation_shape[0]
        self.convolutional_layers = self.get_convolutional_layers(input_n_channels)

        conv_out_flat_size = self._get_conv_out(self.convolutional_layers, observation_shape)
        self.representation_layers = self.get_representation_layers(input_n_features=conv_out_flat_size)

        if n_out_actions is None:
            input_n_features = self.config.MODEL_PARAMETER.NEURONS_PER_REPRESENTATION_LAYER[-1]
        else:
            input_n_features = self.config.MODEL_PARAMETER.NEURONS_PER_REPRESENTATION_LAYER[-1] + n_out_actions
        self.linear_layers = self.get_linear_layers(input_n_features=input_n_features)

    def make_recurrent_linear_model(self, observation_shape):
        input_n_features = observation_shape[0]
        self.representation_layers = self.get_representation_layers(input_n_features=input_n_features)

        input_n_features = self.config.MODEL_PARAMETER.NEURONS_PER_REPRESENTATION_LAYER[-1]
        self.recurrent_layers = self.get_recurrent_layers(input_n_features=input_n_features)

        input_n_features = self.config.MODEL_PARAMETER.HIDDEN_SIZE
        self.linear_layers = self.get_linear_layers(input_n_features=input_n_features)

    def make_recurrent_convolutional_model(self, observation_shape):
        input_n_channels = observation_shape[0]
        self.convolutional_layers = self.get_convolutional_layers(input_n_channels)

        conv_out_flat_size = self._get_conv_out(self.convolutional_layers, observation_shape)
        self.representation_layers = self.get_representation_layers(conv_out_flat_size)

        input_n_features = self.config.MODEL_PARAMETER.NEURONS_PER_REPRESENTATION_LAYER[-1]
        self.recurrent_layers = self.get_recurrent_layers(input_n_features=input_n_features)

        input_n_features = self.config.MODEL_PARAMETER.HIDDEN_SIZE
        self.linear_layers = self.get_linear_layers(input_n_features=input_n_features)

    def _forward(self, obs, save_hidden=False):
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float32, device=self.config.DEVICE)

        if isinstance(self.config.MODEL_PARAMETER, ConfigLinearModel):
            x = self.representation_layers(obs)
            x = self.linear_layers(x)

        elif isinstance(self.config.MODEL_PARAMETER, ConfigConvolutionalModel):
            conv_out = self.convolutional_layers(obs)
            conv_out = torch.flatten(conv_out, start_dim=1)
            x = self.representation_layers(conv_out)
            x = self.linear_layers(x)

        elif isinstance(self.config.MODEL_PARAMETER, ConfigRecurrentLinearModel):
            """
            x: [(observations, hiddens)]
                type(x): list
                len(x): 1

            x[0]: (observations, hiddens)
                type(x[0]): tuple
                len(x[0]): 2

            rnn_in = x[0][0]: observations
                type(rnn_in): ndarray or tensor
                rnn_in.shape: (batch_size, observation_shape)

            h_0 = x[0][1]: hiddens
                type(h_0): tensor
                h_0.shape: torch.Size([num_layers, batch_size, hidden_size])

            rnn_in = rnn_in.unsqueeze(1):
                type(rnn_in): tensor
                rnn_in.shape: (batch_size, 1, observation_shape)

                rnn_in.shape is torch.Size[batch_size, observation_shape],
                but rnn_in.shape must be torch.Size[batch_size, sequence_length, observation_shape]
                Always sequence_length is 1,
                therefore rnn_in.shape is torch.Size[batch_size, 1, observation_shape]

            rnn_out, h_n = self.recurrent_layers(rnn_in, h_0):
                rnn_out.shape: torch.Size(batch_size, 1, hidden_size)
                h_n.shape: torch.Size[num_layers, batch_size, hiddens_size]
            """

            # input
            obs, h_in = obs[0]
            if isinstance(obs, np.ndarray):
                obs = torch.tensor(obs, dtype=torch.float32, device=self.config.DEVICE)
            if isinstance(h_in, np.ndarray):
                h_in = torch.tensor(h_in, dtype=torch.float32, device=self.config.DEVICE)

            # representation layers
            rnn_in = self.representation_layers(obs)

            # recurrent layers
            if rnn_in.ndim == 2:
                rnn_in = rnn_in.unsqueeze(1)
            rnn_out, h_out = self.recurrent_layers(rnn_in, h_in)
            if save_hidden:
                self.recurrent_hidden = h_out.detach()  # save hidden

            # linear layers
            rnn_out_flattened = torch.flatten(rnn_out, start_dim=1)
            x = self.linear_layers(rnn_out_flattened)

        elif isinstance(self.config.MODEL_PARAMETER, ConfigRecurrentConvolutionalModel):
            # input
            obs, h_in = obs[0]
            if isinstance(obs, np.ndarray):
                obs = torch.tensor(obs, dtype=torch.float32, device=self.config.DEVICE)
            if isinstance(h_in, np.ndarray):
                h_in = torch.tensor(h_in, dtype=torch.float32, device=self.config.DEVICE)

            # convolutional layers
            conv_out = self.convolutional_layers(obs)
            conv_out = torch.flatten(conv_out, start_dim=1)

            # representation layers
            x = self.representation_layers(conv_out)

            # recurrent layers
            if x.ndim == 2:
                x = x.unsqueeze(1)
            x, h_out = self.recurrent_layers(x, h_in)
            self.recurrent_hidden = h_out.detach()  # save hidden

            # linear layers
            x = torch.flatten(x, start_dim=1)
            x = self.linear_layers(x)

        else:
            raise ValueError()

        return x
