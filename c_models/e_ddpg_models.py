from typing import Tuple
import numpy as np
import torch
from torch import nn

from a_configuration.b_base.c_models.convolutional_models import ParameterConvolutionalModel
from a_configuration.b_base.c_models.linear_models import ParameterLinearModel
from a_configuration.b_base.c_models.recurrent_convolutional_models import ParameterRecurrentConvolutionalModel
from a_configuration.b_base.c_models.recurrent_linear_models import ParameterRecurrentLinearModel
from c_models.a_models import Model
from c_models.c_policy_models import DiscreteActorModel, ContinuousActorModel, PolicyModel


class ContinuousDdpgActorModel(PolicyModel):
    def __init__(
            self, observation_shape: Tuple[int], n_out_actions: int, parameter=None
    ):
        super(ContinuousDdpgActorModel, self).__init__(
            observation_shape=observation_shape, n_out_actions=n_out_actions, parameter=parameter
        )

        self.mu = nn.Sequential(
            nn.Linear(self.parameter.MODEL.NEURONS_PER_FULLY_CONNECTED_LAYER[-1], self.n_out_actions),
            nn.Tanh()
        )
        self.actor_params += list(self.mu.parameters())

    def pi(self, x):
        x = self.forward_actor(x)
        mu_v = self.mu(x)
        return mu_v


class DdpgCriticModel(Model):
    def __init__(
            self, observation_shape: Tuple[int], n_out_actions: int, n_discrete_actions=None, parameter=None
    ):
        super(DdpgCriticModel, self).__init__(observation_shape, n_out_actions, n_discrete_actions, parameter)
        #######################
        # CRITIC MODEL: BEGIN #
        #######################
        self.critic_params = []
        if isinstance(self.parameter.MODEL, ParameterLinearModel):
            input_n_features = self.observation_shape[0]
            self.critic_fc_layers = self.get_linear_layers(input_n_features=input_n_features + self.n_out_actions)
            self.critic_params += list(self.critic_fc_layers.parameters())

        elif isinstance(self.parameter.MODEL, ParameterConvolutionalModel):
            input_n_channels = self.observation_shape[0]
            self.critic_conv_layers = self.get_conv_layers(input_n_channels=input_n_channels)
            self.critic_params += list(self.critic_conv_layers.parameters())

            conv_out_flat_size = self._get_conv_out(self.conv_layers, self.observation_shape)
            input_n_features = conv_out_flat_size + self.n_out_actions
            self.critic_fc_layers = self.get_linear_layers(input_n_features=input_n_features)
            self.critic_params += list(self.critic_fc_layers.parameters())

        elif isinstance(self.parameter.MODEL, ParameterRecurrentLinearModel):
            input_n_features = self.observation_shape[0] + self.n_out_actions
            self.critic_recurrent_layers = self.get_recurrent_layers(input_n_features=input_n_features)
            self.critic_params += list(self.critic_recurrent_layers.parameters())

            self.critic_fc_layers = self.get_linear_layers(self.parameter.MODEL.HIDDEN_SIZE)
            self.critic_params += list(self.critic_fc_layers.parameters())

        elif isinstance(self.parameter.MODEL, ParameterRecurrentConvolutionalModel):
            input_n_channels = self.observation_shape[0]
            self.critic_conv_layers = self.get_conv_layers(input_n_channels=input_n_channels)
            self.critic_params += list(self.critic_conv_layers.parameters())

            conv_out_flat_size = self._get_conv_out(self.critic_conv_layers, self.observation_shape)
            input_n_features = conv_out_flat_size + self.n_out_actions
            self.critic_fc_layers_1 = nn.Linear(
                in_features=input_n_features, out_features=self.parameter.MODEL.HIDDEN_SIZE
            )
            self.critic_params += list(self.critic_fc_layers_1.parameters())

            self.critic_recurrent_layers = self.get_recurrent_layers(self.parameter.MODEL.HIDDEN_SIZE)
            self.critic_params += list(self.critic_recurrent_layers.parameters())

            self.critic_fc_layers_2 = self.get_linear_layers(self.parameter.MODEL.HIDDEN_SIZE)
            self.critic_params += list(self.critic_fc_layers_2.parameters())
        else:
            raise ValueError()

        self.critic_fc_last_layer = nn.Linear(self.parameter.MODEL.NEURONS_PER_FULLY_CONNECTED_LAYER[-1], 1)
        self.critic_params += list(self.critic_fc_last_layer.parameters())
        #####################
        # CRITIC MODEL: END #
        #####################

    def forward_critic(self, obs, act):
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float32, device=self.parameter.DEVICE)

        if isinstance(act, np.ndarray):
            act = torch.tensor(act, dtype=torch.float32, device=self.parameter.DEVICE)

        if isinstance(self.parameter.MODEL, ParameterLinearModel):
            x = self.critic_fc_layers(torch.cat([obs, act], dim=-1))
        elif isinstance(self.parameter.MODEL, ParameterConvolutionalModel):
            conv_out = self.critic_conv_layers(obs)
            conv_out = torch.flatten(conv_out, start_dim=1)
            x = self.critic_fc_layers(torch.cat([conv_out, act], dim=-1))
        elif isinstance(self.parameter.MODEL, ParameterRecurrentLinearModel):
            rnn_in, hidden_in = obs[0]
            if isinstance(rnn_in, np.ndarray):
                rnn_in = torch.tensor(rnn_in, dtype=torch.float32, device=self.parameter.DEVICE)
            if isinstance(hidden_in, np.ndarray):
                hidden_in = torch.tensor(hidden_in, dtype=torch.float32, device=self.parameter.DEVICE)

            if act.ndim == 2:
                act = act.unsqueeze(1)

            if rnn_in.ndim == 2:
                rnn_in = rnn_in.unsqueeze(1)

            rnn_out, hidden_out = self.critic_recurrent_layers(torch.cat([rnn_in, act], dim=-1), hidden_in)
            self.recurrent_hidden = hidden_out.detach()  # save hidden
            rnn_out_flattened = torch.flatten(rnn_out, start_dim=1)
            x = self.critic_fc_layers(rnn_out_flattened)

        elif isinstance(self.parameter.MODEL, ParameterRecurrentConvolutionalModel):
            x, hidden_in = obs[0]
            if isinstance(x, np.ndarray):
                x = torch.tensor(x, dtype=torch.float32, device=self.parameter.DEVICE)
            if isinstance(hidden_in, np.ndarray):
                hidden_in = torch.tensor(hidden_in, dtype=torch.float32, device=self.parameter.DEVICE)

            conv_out = self.critic_conv_layers(x)
            conv_out = torch.flatten(conv_out, start_dim=1)
            x = self.critic_fc_layers_1(conv_out)

            rnn_in = x
            if act.ndim == 2:
                act = act.unsqueeze(1)

            if rnn_in.ndim == 2:
                rnn_in = rnn_in.unsqueeze(1)

            rnn_out, hidden_out = self.critic_recurrent_layers(torch.cat([rnn_in, act], dim=-1), hidden_in)
            self.recurrent_hidden = hidden_out.detach()  # save hidden
            rnn_out_flattened = torch.flatten(rnn_out, start_dim=1)
            x = self.critic_fc_layers_2(rnn_out_flattened)

        else:
            raise ValueError()
        return x

    def q(self, obs, act):
        x = self.forward_critic(obs, act)
        q_value = self.critic_fc_last_layer(x)
        return q_value


class DiscreteDdpgModel:
    def __init__(
            self, observation_shape: Tuple[int], n_out_actions: int, n_discrete_actions=None, parameter=None
    ):
        self.parameter = parameter

        self.actor_model = DiscreteActorModel(
            observation_shape=observation_shape, n_out_actions=n_out_actions, n_discrete_actions=n_discrete_actions,
            parameter=self.parameter
        ).to(self.parameter.DEVICE)

        self.critic_model = DdpgCriticModel(
            observation_shape=observation_shape, n_out_actions=n_out_actions, n_discrete_actions=n_discrete_actions,
            parameter=self.parameter
        ).to(self.parameter.DEVICE)


class ContinuousDdpgModel:
    def __init__(
            self, observation_shape: Tuple[int], n_out_actions: int, parameter=None
    ):
        self.parameter = parameter

        self.actor_model = ContinuousDdpgActorModel(
            observation_shape=observation_shape, n_out_actions=n_out_actions, parameter=self.parameter
        ).to(self.parameter.DEVICE)

        self.critic_model = DdpgCriticModel(
            observation_shape=observation_shape, n_out_actions=n_out_actions, n_discrete_actions=None,
            parameter=self.parameter
        ).to(self.parameter.DEVICE)
