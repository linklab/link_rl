from typing import Tuple
import numpy as np
import torch
from torch import nn

from a_configuration.a_base_config.c_models.convolutional_models import ConfigConvolutionalModel
from a_configuration.a_base_config.c_models.linear_models import ConfigLinearModel
from a_configuration.a_base_config.c_models.recurrent_convolutional_models import ConfigRecurrentConvolutionalModel
from a_configuration.a_base_config.c_models.recurrent_linear_models import ConfigRecurrentLinearModel
from c_models.a_models import Model


class QNet(Model):
    # self.n_out_actions: 1
    # self.n_discrete_actions: 4 (for gridworld)
    def __init__(
            self, observation_shape: Tuple[int], n_out_actions: int, n_discrete_actions=None, config=None
    ):
        super(QNet, self).__init__(observation_shape, n_out_actions, n_discrete_actions, config)

        self.qnet_params = []
        if isinstance(self.config.MODEL_PARAMETER, ConfigLinearModel):
            input_n_features = self.observation_shape[0]
            self.representation_layers = None
            self.fc_layers = self.get_linear_layers(input_n_features)
            self.qnet_params += list(self.fc_layers.parameters())

        elif isinstance(self.config.MODEL_PARAMETER, ConfigConvolutionalModel):
            input_n_channels = self.observation_shape[0]
            self.conv_layers = self.get_conv_layers(input_n_channels)
            self.qnet_params += list(self.conv_layers.parameters())

            conv_out_flat_size = self._get_conv_out(self.conv_layers, observation_shape)
            self.fc_layers = self.get_linear_layers(conv_out_flat_size)
            self.qnet_params += list(self.fc_layers.parameters())

        elif isinstance(self.config.MODEL_PARAMETER, ConfigRecurrentLinearModel):
            input_n_features = self.observation_shape[0]
            self.recurrent_layers = self.get_recurrent_layers(input_n_features)
            self.qnet_params += list(self.recurrent_layers.parameters())

            # recurrent_out_flat_size, _ = self._get_recurrent_out(self.recurrent_layers, input_n_features)
            # self.fc_layers = self.get_linear_layers(recurrent_out_flat_size)

            self.fc_layers = self.get_linear_layers(self.config.MODEL_PARAMETER.HIDDEN_SIZE)
            self.qnet_params += list(self.fc_layers.parameters())

        elif isinstance(self.config.MODEL_PARAMETER, ConfigRecurrentConvolutionalModel):
            input_n_channels = self.observation_shape[0]
            self.conv_layers = self.get_conv_layers(input_n_channels)
            self.qnet_params += list(self.conv_layers.parameters())

            conv_out_flat_size = self._get_conv_out(self.conv_layers, observation_shape)
            self.fc_layers_1 = nn.Linear(conv_out_flat_size, self.config.MODEL_PARAMETER.HIDDEN_SIZE)
            self.qnet_params += list(self.fc_layers_1.parameters())

            self.recurrent_layers = self.get_recurrent_layers(self.config.MODEL_PARAMETER.HIDDEN_SIZE)
            self.qnet_params += list(self.recurrent_layers.parameters())

            # recurrent_out_flat_size, _ = self._get_recurrent_out(
            #     self.recurrent_layers,
            #     self.config.MODEL_PARAMETER.HIDDEN_SIZE
            # )
            self.fc_layers_2 = self.get_linear_layers(self.config.MODEL_PARAMETER.HIDDEN_SIZE)
            self.qnet_params += list(self.fc_layers_2.parameters())

        else:
            raise ValueError()

        self.fc_last = nn.Linear(
            self.config.MODEL_PARAMETER.NEURONS_PER_FULLY_CONNECTED_LAYER[-1], self.n_discrete_actions
        )
        self.qnet_params += list(self.fc_last.parameters())

        self.version = 0

    def forward(self, x, save_hidden=False):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32, device=self.config.DEVICE)

        if isinstance(self.config.MODEL_PARAMETER, ConfigLinearModel):
            x = self.recurrent_layers(x)
            x = self.fc_layers(x)
        elif isinstance(self.config.MODEL_PARAMETER, ConfigConvolutionalModel):
            # print("x.shape:", x.shape)
            conv_out = self.conv_layers(x)
            conv_out = torch.flatten(conv_out, start_dim=1)
            x = self.fc_layers(conv_out)
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

            rnn_in, h_0 = x[0]
            if isinstance(rnn_in, np.ndarray):
                rnn_in = torch.tensor(rnn_in, dtype=torch.float32, device=self.config.DEVICE)
            if isinstance(h_0, np.ndarray):
                h_0 = torch.tensor(h_0, dtype=torch.float32, device=self.config.DEVICE)

            if rnn_in.ndim == 2:
                rnn_in = rnn_in.unsqueeze(1)

            rnn_out, h_n = self.recurrent_layers(rnn_in, h_0)

            if save_hidden:
                self.recurrent_hidden = h_n.detach()  # save hidden

            rnn_out_flattened = torch.flatten(rnn_out, start_dim=1)

            x = self.fc_layers(rnn_out_flattened)

        elif isinstance(self.config.MODEL_PARAMETER, ConfigRecurrentConvolutionalModel):
            x, h_0 = x[0]
            if isinstance(x, np.ndarray):
                x = torch.tensor(x, dtype=torch.float32, device=self.config.DEVICE)
            if isinstance(h_0, np.ndarray):
                h_0 = torch.tensor(h_0, dtype=torch.float32, device=self.config.DEVICE)

            conv_out = self.conv_layers(x)
            conv_out = torch.flatten(conv_out, start_dim=1)
            x = self.fc_layers_1(conv_out)

            rnn_in = x
            if rnn_in.ndim == 2:
                rnn_in = rnn_in.unsqueeze(0)

            rnn_out, h_n = self.recurrent_layers(rnn_in, h_0)
            self.recurrent_hidden = h_n.detach()  # save hidden
            rnn_out = torch.flatten(rnn_out, start_dim=1)
            x = self.fc_layers_2(rnn_out)

        else:
            raise ValueError()

        x = self.fc_last(x)
        return x


class DuelingQNet(Model):
    # self.n_out_actions: 1
    # self.n_discrete_actions: 4 (for gridworld)
    def __init__(
            self, observation_shape: Tuple[int], n_out_actions: int, n_discrete_actions=None, config=None
    ):
        super(DuelingQNet, self).__init__(observation_shape, n_out_actions, n_discrete_actions, config)

        self.qnet_params = []
        if isinstance(self.config.MODEL_PARAMETER, ConfigLinearModel):
            input_n_features = self.observation_shape[0]
            self.fc_layers = self.get_linear_layers(input_n_features=input_n_features)
            self.qnet_params += list(self.fc_layers.parameters())
        elif isinstance(self.config.MODEL_PARAMETER, ConfigConvolutionalModel):
            input_n_channels = self.observation_shape[0]
            self.conv_layers = self.get_conv_layers(input_n_channels=input_n_channels)
            self.qnet_params += list(self.conv_layers.parameters())
            conv_out_flat_size = self._get_conv_out(self.conv_layers, observation_shape)
            self.fc_layers = self.get_linear_layers(input_n_features=conv_out_flat_size)
            self.qnet_params += list(self.fc_layers.parameters())
        else:
            raise ValueError()

        self.fc_last_adv = nn.Linear(
            self.config.MODEL_PARAMETER.NEURONS_PER_FULLY_CONNECTED_LAYER[-1], self.n_discrete_actions
        )
        self.qnet_params += list(self.fc_last_adv.parameters())

        self.fc_last_val = nn.Linear(
            self.config.MODEL_PARAMETER.NEURONS_PER_FULLY_CONNECTED_LAYER[-1], 1
        )
        self.qnet_params += list(self.fc_last_val.parameters())

        self.version = 0

    def forward(self, x, save_hidden=False):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32, device=self.config.DEVICE)

        if isinstance(self.config.MODEL_PARAMETER, ConfigLinearModel):
            x = self.fc_layers(x)
            adv = self.fc_last_adv(x)
            val = self.fc_last_val(x)
        elif isinstance(self.config.MODEL_PARAMETER, ConfigConvolutionalModel):
            conv_out = self.conv_layers(x)
            conv_out = torch.flatten(conv_out, start_dim=1)
            x = self.fc_layers(conv_out)
            adv = self.fc_last_adv(x)
            val = self.fc_last_val(x)
        else:
            raise ValueError()

        q_values = val + adv - torch.mean(adv, dim=-1, keepdim=True)
        return q_values
