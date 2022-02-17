from abc import abstractmethod

import torch
from torch import nn
from typing import Tuple
from collections import OrderedDict
import numpy as np

from a_configuration.a_base_config.c_models.config_convolutional_models import ConfigConvolutionalModel
from a_configuration.a_base_config.c_models.config_linear_models import ConfigLinearModel
from a_configuration.a_base_config.c_models.config_recurrent_convolutional_models import ConfigRecurrentConvolutionalModel
from a_configuration.a_base_config.c_models.config_recurrent_linear_models import ConfigRecurrentLinearModel


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

        self.convolutional_layers = None
        self.representation_layers = None
        self.recurrent_layers = None
        self.linear_layers = None

    def init_recurrent_hidden(self):
        if self.is_recurrent_model:
            self.recurrent_hidden = torch.zeros(
                self.config.MODEL_PARAMETER.NUM_LAYERS,
                1,  # batch_size is always 1
                self.config.MODEL_PARAMETER.HIDDEN_SIZE,
                dtype=torch.float32,
                device=self.config.DEVICE
            )

    ####################
    #   get_*_layers   #
    ####################

    def get_linear_layers(self, input_n_features, activation=nn.ReLU()):
        if not self.config.MODEL_PARAMETER.NEURONS_PER_FULLY_CONNECTED_LAYER:
            print("self.config.MODEL_PARAMETER.NEURONS_PER_FULLY_CONNECTED_LAYER is empty")
            return nn.Sequential()

        linear_layers_dict = OrderedDict()
        neurons_per_fully_connected_layer = self.config.MODEL_PARAMETER.NEURONS_PER_FULLY_CONNECTED_LAYER.copy()
        neurons_per_fully_connected_layer.append(input_n_features)
        # neurons_per_fully_connected_layer[-1] == input_n_features

        for idx in range(len(self.config.MODEL_PARAMETER.NEURONS_PER_FULLY_CONNECTED_LAYER)):
            # Linear Layer
            linear_layers_dict["linear_{0}".format(idx)] = nn.Linear(
                neurons_per_fully_connected_layer[idx - 1],
                neurons_per_fully_connected_layer[idx]
            )

            # Layer Normalization
            if self.config.USE_LAYER_NORM:
                linear_layers_dict["linear_{0}_norm".format(idx)] = nn.LayerNorm(
                    neurons_per_fully_connected_layer[idx]
                )

            # Activation Function
            linear_layers_dict["linear_{0}_activation".format(idx)] = activation

        linear_layers = nn.Sequential(linear_layers_dict)
        return linear_layers

    def get_representation_layers(self, input_n_features, activation=nn.ReLU()):
        if not self.config.MODEL_PARAMETER.NEURONS_PER_REPRESENTATION_LAYER:
            print("self.config.MODEL_PARAMETER.NEURONS_PER_REPRESENTATION_LAYER is empty")
            return nn.Sequential()

        representation_layers_dict = OrderedDict()
        neurons_per_representation_layer = self.config.MODEL_PARAMETER.NEURONS_PER_REPRESENTATION_LAYER.copy()
        neurons_per_representation_layer.append(input_n_features)
        # neurons_per_representation_layer[-1] == input_n_features

        for idx in range(len(self.config.MODEL_PARAMETER.NEURONS_PER_REPRESENTATION_LAYER)):
            # Representation Layer
            representation_layers_dict["representation_{0}".format(idx)] = nn.Linear(
                neurons_per_representation_layer[idx - 1],
                neurons_per_representation_layer[idx]
            )

            # Layer Normalization
            if self.config.USE_LAYER_NORM:
                representation_layers_dict["representation_{0}_norm".format(idx)] = nn.LayerNorm(
                    neurons_per_representation_layer[idx]
                )

            # Activation Function
            representation_layers_dict["representation_{0}_activation".format(idx)] = activation

        representation_layers = nn.Sequential(representation_layers_dict)
        return representation_layers

    def get_convolutional_layers(self, input_n_channels, activation=nn.ReLU()):
        assert self.config.MODEL_PARAMETER.OUT_CHANNELS_PER_LAYER
        assert self.config.MODEL_PARAMETER.KERNEL_SIZE_PER_LAYER
        assert self.config.MODEL_PARAMETER.STRIDE_PER_LAYER

        convolutional_layers_dict = OrderedDict()
        out_channels_per_layer = self.config.MODEL_PARAMETER.OUT_CHANNELS_PER_LAYER.copy()
        out_channels_per_layer.append(input_n_channels)
        # out_channels_per_layer[-1] == input_n_channels

        for idx in range(len(self.config.MODEL_PARAMETER.OUT_CHANNELS_PER_LAYER)):
            # Convolutional Layer
            convolutional_layers_dict["conv_{0}".format(idx)] = nn.Conv2d(
                in_channels=out_channels_per_layer[idx - 1],
                out_channels=out_channels_per_layer[idx],
                kernel_size=self.config.MODEL_PARAMETER.KERNEL_SIZE_PER_LAYER[idx],
                stride=self.config.MODEL_PARAMETER.STRIDE_PER_LAYER[idx],
                padding=self.config.MODEL_PARAMETER.PADDING
            )

            # Activation Function
            convolutional_layers_dict["conv_{0}_activation".format(idx)] = activation

        convolutional_layers = nn.Sequential(convolutional_layers_dict)
        return convolutional_layers

    def _get_conv_out(self, conv_layers, shape):
        conv_out = conv_layers(torch.zeros(1, *shape))
        return int(np.prod(conv_out.size()))

    # def get_layer_normalization(self, layer_dict, layer_idx):
    #     if isinstance(self.config.MODEL_PARAMETER, ConfigLinearModel):
    #         linear_layers_dict = layer_dict
    #         linear_layers_dict["linear_{0}_norm".format(layer_idx)] = nn.LayerNorm(
    #             self.config.MODEL_PARAMETER.NEURONS_PER_FULLY_CONNECTED_LAYER[layer_idx]
    #         )
    #     elif isinstance(self.config.MODEL_PARAMETER, ConfigConvolutionalModel):
    #         pass
    #     elif isinstance(self.config.MODEL_PARAMETER, ConfigRecurrentLinearModel):
    #         pass

    def get_recurrent_layers(self, input_n_features):
        assert self.config.MODEL_PARAMETER.HIDDEN_SIZE
        assert self.config.MODEL_PARAMETER.NUM_LAYERS

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

    ####################
    #   make_*_model   #
    ####################

    def make_linear_model(self, observation_shape, n_out_actions=None, activation=nn.ReLU()):
        # representation_layers
        input_n_features = observation_shape[0]
        self.representation_layers = self.get_representation_layers(
            input_n_features=input_n_features, activation=activation
        )

        # linear_layers
        if self.config.MODEL_PARAMETER.NEURONS_PER_REPRESENTATION_LAYER:
            input_n_features = self.config.MODEL_PARAMETER.NEURONS_PER_REPRESENTATION_LAYER[-1]

        if n_out_actions is not None:
            input_n_features = input_n_features + n_out_actions

        self.linear_layers = self.get_linear_layers(input_n_features=input_n_features, activation=activation)

    def make_convolutional_model(self, observation_shape, n_out_actions=None, activation=nn.ReLU()):
        # convolutional_layers
        input_n_channels = observation_shape[0]
        self.convolutional_layers = self.get_convolutional_layers(input_n_channels, activation=activation)

        # representation_layers
        conv_out_flat_size = self._get_conv_out(self.convolutional_layers, observation_shape)
        self.representation_layers = self.get_representation_layers(
            input_n_features=conv_out_flat_size, activation=activation
        )

        # linear_layers
        if self.config.MODEL_PARAMETER.NEURONS_PER_REPRESENTATION_LAYER:
            input_n_features = self.config.MODEL_PARAMETER.NEURONS_PER_REPRESENTATION_LAYER[-1]
        else:
            input_n_features = conv_out_flat_size

        if n_out_actions is not None:
            input_n_features = input_n_features + n_out_actions

        self.linear_layers = self.get_linear_layers(input_n_features=input_n_features, activation=activation)

    def make_recurrent_linear_model(self, observation_shape, activation=nn.ReLU()):
        # representation_layers
        input_n_features = observation_shape[0]
        self.representation_layers = self.get_representation_layers(
            input_n_features=input_n_features, activation=activation
        )

        # recurrent_layers
        if self.config.MODEL_PARAMETER.NEURONS_PER_REPRESENTATION_LAYER:
            input_n_features = self.config.MODEL_PARAMETER.NEURONS_PER_REPRESENTATION_LAYER[-1]
        self.recurrent_layers = self.get_recurrent_layers(input_n_features=input_n_features)

        # linear_layers
        input_n_features = self.config.MODEL_PARAMETER.HIDDEN_SIZE
        self.linear_layers = self.get_linear_layers(input_n_features=input_n_features, activation=activation)

    def make_recurrent_convolutional_model(self, observation_shape, activation=nn.ReLU()):
        # convolutional_layers
        input_n_channels = observation_shape[0]
        self.convolutional_layers = self.get_convolutional_layers(input_n_channels, activation=activation)

        # representation_layers
        conv_out_flat_size = self._get_conv_out(self.convolutional_layers, observation_shape)
        self.representation_layers = self.get_representation_layers(conv_out_flat_size, activation=activation)

        # recurrent_layers
        if self.config.MODEL_PARAMETER.NEURONS_PER_REPRESENTATION_LAYER:
            input_n_features = self.config.MODEL_PARAMETER.NEURONS_PER_REPRESENTATION_LAYER[-1]
        else:
            input_n_features = conv_out_flat_size
        self.recurrent_layers = self.get_recurrent_layers(input_n_features=input_n_features)

        # linear_layers
        input_n_features = self.config.MODEL_PARAMETER.HIDDEN_SIZE
        self.linear_layers = self.get_linear_layers(input_n_features=input_n_features, activation=activation)

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
            # print("obs[0].shape:", obs[0].shape)
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

    def _get_forward_pre_out(self, observation_shape):
        obs = torch.zeros(1, *observation_shape)
        if self.is_recurrent_model:
            obs = [(obs, self.recurrent_hidden.cpu())]

        forward_pre_out = self._forward(obs)

        return int(np.prod(forward_pre_out.size()))
