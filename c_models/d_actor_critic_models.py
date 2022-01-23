from typing import Tuple
import numpy as np
import torch
from torch import nn

from a_configuration.b_base.c_models.convolutional_models import ParameterConvolutionalModel
from a_configuration.b_base.c_models.linear_models import ParameterLinearModel
from a_configuration.b_base.c_models.recurrent_convolutional_models import ParameterRecurrentConvolutionalModel
from a_configuration.b_base.c_models.recurrent_linear_models import ParameterRecurrentLinearModel
from c_models.a_models import Model
from c_models.c_policy_models import DiscreteActorModel, ContinuousActorModel


class CriticModel(Model):
    def __init__(
            self, observation_shape: Tuple[int], n_out_actions: int, n_discrete_actions=None, parameter=None
    ):
        super(CriticModel, self).__init__(observation_shape, n_out_actions, n_discrete_actions, parameter)

        #######################
        # CRITIC MODEL: BEGIN #
        #######################
        self.critic_params = []
        if isinstance(self.parameter.MODEL_PARAMETER, ParameterLinearModel):
            input_n_features = self.observation_shape[0]
            self.critic_fc_layers = self.get_linear_layers(input_n_features=input_n_features)
            self.critic_params += list(self.critic_fc_layers.parameters())

        elif isinstance(self.parameter.MODEL_PARAMETER, ParameterConvolutionalModel):
            input_n_channels = self.observation_shape[0]
            self.critic_conv_layers = self.get_conv_layers(input_n_channels=input_n_channels)
            self.critic_params += list(self.critic_conv_layers.parameters())

            conv_out_flat_size = self._get_conv_out(self.critic_conv_layers, observation_shape)
            self.critic_fc_layers = self.get_linear_layers(input_n_features=conv_out_flat_size)
            self.critic_params += list(self.critic_fc_layers.parameters())

        elif isinstance(self.parameter.MODEL_PARAMETER, ParameterRecurrentLinearModel):
            input_n_features = self.observation_shape[0]
            self.critic_recurrent_layers = self.get_recurrent_layers(input_n_features=input_n_features)
            self.critic_params += list(self.critic_recurrent_layers.parameters())

            self.critic_fc_layers = self.get_linear_layers(self.parameter.MODEL_PARAMETER.HIDDEN_SIZE)
            self.critic_params += list(self.critic_fc_layers.parameters())

        elif isinstance(self.parameter.MODEL_PARAMETER, ParameterRecurrentConvolutionalModel):
            input_n_channels = self.observation_shape[0]
            self.critic_conv_layers = self.get_conv_layers(input_n_channels=input_n_channels)
            self.critic_params += list(self.critic_conv_layers.parameters())

            conv_out_flat_size = self._get_conv_out(self.critic_conv_layers, self.observation_shape)
            self.critic_fc_layers_1 = nn.Linear(
                in_features=conv_out_flat_size, out_features=self.parameter.MODEL_PARAMETER.HIDDEN_SIZE
            )
            self.critic_params += list(self.critic_fc_layers_1.parameters())

            self.critic_recurrent_layers = self.get_recurrent_layers(self.parameter.MODEL_PARAMETER.HIDDEN_SIZE)
            self.critic_params += list(self.critic_recurrent_layers.parameters())

            self.critic_fc_layers_2 = self.get_linear_layers(self.parameter.MODEL_PARAMETER.HIDDEN_SIZE)
            self.critic_params += list(self.critic_fc_layers_2.parameters())
        else:
            raise ValueError()

        self.critic_fc_last = nn.Linear(self.parameter.MODEL_PARAMETER.NEURONS_PER_FULLY_CONNECTED_LAYER[-1], 1)
        self.critic_params += list(self.critic_fc_last.parameters())
        #####################
        # CRITIC MODEL: END #
        #####################

    def forward_critic(self, obs):
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float32, device=self.parameter.DEVICE)

        if isinstance(self.parameter.MODEL_PARAMETER, ParameterLinearModel):
            x = self.critic_fc_layers(obs)

        elif isinstance(self.parameter.MODEL_PARAMETER, ParameterConvolutionalModel):
            conv_out = self.critic_conv_layers(obs)
            conv_out = torch.flatten(conv_out, start_dim=1)
            x = self.critic_fc_layers(conv_out)

        elif isinstance(self.parameter.MODEL_PARAMETER, ParameterRecurrentLinearModel):
            rnn_in, h_0 = obs[0]
            if isinstance(rnn_in, np.ndarray):
                rnn_in = torch.tensor(rnn_in, dtype=torch.float32, device=self.parameter.DEVICE)
            if isinstance(h_0, np.ndarray):
                h_0 = torch.tensor(h_0, dtype=torch.float32, device=self.parameter.DEVICE)
            if rnn_in.ndim == 2:
                rnn_in = rnn_in.unsqueeze(1)

            rnn_out, h_n = self.critic_recurrent_layers(rnn_in, h_0)
            self.recurrent_hidden = h_n.detach()  # save hidden
            rnn_out_flattened = torch.flatten(rnn_out, start_dim=1)

            x = self.critic_fc_layers(rnn_out_flattened)

        elif isinstance(self.parameter.MODEL_PARAMETER, ParameterRecurrentConvolutionalModel):
            x, h_0 = obs[0]
            if isinstance(x, np.ndarray):
                x = torch.tensor(x, dtype=torch.float32, device=self.parameter.DEVICE)
            if isinstance(h_0, np.ndarray):
                h_0 = torch.tensor(h_0, dtype=torch.float32, device=self.parameter.DEVICE)

            conv_out = self.critic_conv_layers(x)
            conv_out = torch.flatten(conv_out, start_dim=1)
            x = self.critic_fc_layers_1(conv_out)

            rnn_in = x
            if rnn_in.ndim == 2:
                rnn_in = rnn_in.unsqueeze(1)

            rnn_out, h_n = self.critic_recurrent_layers(rnn_in, h_0)
            self.recurrent_hidden = h_n.detach()  # save hidden
            rnn_out = torch.flatten(rnn_out, start_dim=1)
            x = self.critic_fc_layers_2(rnn_out)

        else:
            raise ValueError()
        return x

    def v(self, x):
        x = self.forward_critic(x)
        v = self.critic_fc_last(x)
        return v


class DiscreteActorCriticModel:
    def __init__(
            self, observation_shape: Tuple[int], n_out_actions: int, n_discrete_actions=None, parameter=None
    ):
        self.parameter = parameter

        self.actor_model = DiscreteActorModel(
            observation_shape=observation_shape, n_out_actions=n_out_actions, n_discrete_actions=n_discrete_actions,
            parameter=self.parameter
        ).to(self.parameter.DEVICE)

        self.critic_model = CriticModel(
            observation_shape=observation_shape, n_out_actions=n_out_actions, n_discrete_actions=n_discrete_actions,
            parameter=self.parameter
        ).to(self.parameter.DEVICE)


class ContinuousActorCriticModel:
    def __init__(
            self, observation_shape: Tuple[int], n_out_actions: int, parameter=None
    ):
        self.parameter = parameter

        self.actor_model = ContinuousActorModel(
            observation_shape=observation_shape, n_out_actions=n_out_actions, parameter=parameter
        ).to(self.parameter.DEVICE)

        self.critic_model = CriticModel(
            observation_shape=observation_shape, n_out_actions=n_out_actions, n_discrete_actions=None,
            parameter=parameter
        ).to(self.parameter.DEVICE)
