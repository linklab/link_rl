from typing import Tuple
import numpy as np
import torch
from torch import nn

from a_configuration.b_base.c_models.convolutional_models import ParameterConvolutionalModel
from a_configuration.b_base.c_models.linear_models import ParameterLinearModel
from a_configuration.b_base.c_models.recurrent_convolutional_models import ParameterRecurrentConvolutionalModel
from a_configuration.b_base.c_models.recurrent_linear_models import ParameterRecurrentLinearModel
from c_models.a_models import Model


class QNet(Model):
    # self.n_out_actions: 1
    # self.n_discrete_actions: 4 (for gridworld)
    def __init__(
            self, observation_shape: Tuple[int], n_out_actions: int, n_discrete_actions=None, parameter=None
    ):
        super(QNet, self).__init__(observation_shape, n_out_actions, n_discrete_actions, parameter)

        self.qnet_params = []
        if isinstance(self.parameter.MODEL, ParameterLinearModel):
            input_n_features = self.observation_shape[0]
            self.fc_layers = self.get_linear_layers(input_n_features)
            self.qnet_params += list(self.fc_layers.parameters())

        elif isinstance(self.parameter.MODEL, ParameterConvolutionalModel):
            input_n_channels = self.observation_shape[0]
            self.conv_layers = self.get_conv_layers(input_n_channels)
            self.qnet_params += list(self.conv_layers.parameters())

            conv_out_flat_size = self._get_conv_out(self.conv_layers, observation_shape)
            self.fc_layers = self.get_linear_layers(conv_out_flat_size)
            self.qnet_params += list(self.fc_layers.parameters())

        elif isinstance(self.parameter.MODEL, ParameterRecurrentLinearModel):
            input_n_features = self.observation_shape[0]
            self.recurrent_layers = self.get_recurrent_layers(input_n_features)
            self.qnet_params += list(self.recurrent_layers.parameters())

            #recurrent_out_flat_size, _ = self._get_recurrent_out(self.recurrent_layers, input_n_features)
            #self.fc_layers = self.get_linear_layers(recurrent_out_flat_size)

            self.fc_layers = self.get_linear_layers(self.parameter.MODEL.HIDDEN_SIZE)
            self.qnet_params += list(self.fc_layers.parameters())

        elif isinstance(self.parameter.MODEL, ParameterRecurrentConvolutionalModel):
            input_n_channels = self.observation_shape[0]
            self.conv_layers = self.get_conv_layers(input_n_channels)
            self.qnet_params += list(self.conv_layers.parameters())

            conv_out_flat_size = self._get_conv_out(self.conv_layers, observation_shape)
            self.fc_layers_1 = nn.Linear(conv_out_flat_size, self.parameter.MODEL.HIDDEN_SIZE)
            self.qnet_params += list(self.fc_layers_1.parameters())

            self.recurrent_layers = self.get_recurrent_layers(self.parameter.MODEL.HIDDEN_SIZE)
            self.qnet_params += list(self.recurrent_layers.parameters())

            # recurrent_out_flat_size, _ = self._get_recurrent_out(
            #     self.recurrent_layers,
            #     self.parameter.MODEL.HIDDEN_SIZE
            # )
            self.fc_layers_2 = self.get_linear_layers(self.parameter.MODEL.HIDDEN_SIZE)
            self.qnet_params += list(self.fc_layers_2.parameters())

        else:
            raise ValueError()

        self.fc_last = nn.Linear(
            self.parameter.MODEL.NEURONS_PER_FULLY_CONNECTED_LAYER[-1], self.n_discrete_actions
        )
        self.qnet_params += list(self.fc_last.parameters())

        self.version = 0

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32, device=self.parameter.DEVICE)

        if isinstance(self.parameter.MODEL, ParameterLinearModel):
            x = self.fc_layers(x)
        elif isinstance(self.parameter.MODEL, ParameterConvolutionalModel):
            #print("x.shape:", x.shape)
            conv_out = self.conv_layers(x)
            conv_out = torch.flatten(conv_out, start_dim=1)
            x = self.fc_layers(conv_out)
        elif isinstance(self.parameter.MODEL, ParameterRecurrentLinearModel):
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
                rnn_in = torch.tensor(rnn_in, dtype=torch.float32, device=self.parameter.DEVICE)
            if isinstance(h_0, np.ndarray):
                h_0 = torch.tensor(h_0, dtype=torch.float32, device=self.parameter.DEVICE)

            if rnn_in.ndim == 2:
                rnn_in = rnn_in.unsqueeze(0)

            rnn_out, h_n = self.recurrent_layers(rnn_in, h_0)
            self.recurrent_hidden = h_n.detach()  # save hidden
            # self.init_recurrent_hidden()
            rnn_out_flattened = torch.flatten(rnn_out, start_dim=1)

            #print(rnn_in.shape, rnn_out.shape, rnn_out_flattened.shape, "!!!!!")

            x = self.fc_layers(rnn_out_flattened)

        elif isinstance(self.parameter.MODEL, ParameterRecurrentConvolutionalModel):
            x, h_0 = x[0]
            if isinstance(x, np.ndarray):
                x = torch.tensor(x, dtype=torch.float32, device=self.parameter.DEVICE)
            if isinstance(h_0, np.ndarray):
                h_0 = torch.tensor(h_0, dtype=torch.float32, device=self.parameter.DEVICE)

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
            self, observation_shape: Tuple[int], n_out_actions: int, n_discrete_actions=None, parameter=None
    ):
        super(DuelingQNet, self).__init__(observation_shape, n_out_actions, n_discrete_actions, parameter)

        self.qnet_params = []
        if isinstance(self.parameter.MODEL, ParameterLinearModel):
            input_n_features = self.observation_shape[0]
            self.fc_layers = self.get_linear_layers(input_n_features=input_n_features)
            self.qnet_params += list(self.fc_layers.parameters())
        elif isinstance(self.parameter.MODEL, ParameterConvolutionalModel):
            input_n_channels = self.observation_shape[0]
            self.conv_layers = self.get_conv_layers(input_n_channels=input_n_channels)
            self.qnet_params += list(self.conv_layers.parameters())
            conv_out_flat_size = self._get_conv_out(self.conv_layers, observation_shape)
            self.fc_layers = self.get_linear_layers(input_n_features=conv_out_flat_size)
            self.qnet_params += list(self.fc_layers.parameters())
        else:
            raise ValueError()

        self.fc_last_adv = nn.Linear(
            self.parameter.MODEL.NEURONS_PER_FULLY_CONNECTED_LAYER[-1], self.n_discrete_actions
        )
        self.qnet_params += list(self.fc_last_adv.parameters())

        self.fc_last_val = nn.Linear(
            self.parameter.MODEL.NEURONS_PER_FULLY_CONNECTED_LAYER[-1], 1
        )
        self.qnet_params += list(self.fc_last_val.parameters())

        self.version = 0

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32, device=self.parameter.DEVICE)

        if isinstance(self.parameter.MODEL, ParameterLinearModel):
            x = self.fc_layers(x)
            adv = self.fc_last_adv(x)
            val = self.fc_last_val(x)
        elif isinstance(self.parameter.MODEL, ParameterConvolutionalModel):
            conv_out = self.conv_layers(x)
            conv_out = torch.flatten(conv_out, start_dim=1)
            x = self.fc_layers(conv_out)
            adv = self.fc_last_adv(x)
            val = self.fc_last_val(x)
        else:
            raise ValueError()

        q_values = val + adv - torch.mean(adv, dim=-1, keepdim=True)
        return q_values
