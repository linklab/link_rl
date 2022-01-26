from abc import abstractmethod
from typing import Tuple
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from a_configuration.b_base.c_models.convolutional_models import ParameterConvolutionalModel
from a_configuration.b_base.c_models.linear_models import ParameterLinearModel
from a_configuration.b_base.c_models.recurrent_convolutional_models import ParameterRecurrentConvolutionalModel
from a_configuration.b_base.c_models.recurrent_linear_models import ParameterRecurrentLinearModel
from c_models.a_models import Model


class PolicyModel(Model):
    def __init__(
            self, observation_shape: Tuple[int], n_out_actions: int, n_discrete_actions=None, parameter=None
    ):
        super(PolicyModel, self).__init__(observation_shape, n_out_actions, n_discrete_actions, parameter)

        self.actor_params = []
        if isinstance(self.parameter.MODEL_PARAMETER, ParameterLinearModel):
            input_n_features = self.observation_shape[0]
            self.actor_fc_layers = self.get_linear_layers(input_n_features=input_n_features)
            self.actor_params += list(self.actor_fc_layers.parameters())

        elif isinstance(self.parameter.MODEL_PARAMETER, ParameterConvolutionalModel):
            input_n_channels = self.observation_shape[0]
            self.actor_conv_layers = self.get_conv_layers(input_n_channels=input_n_channels)
            self.actor_params += list(self.actor_conv_layers.parameters())

            conv_out_flat_size = self._get_conv_out(self.actor_conv_layers, self.observation_shape)
            self.actor_fc_layers = self.get_linear_layers(input_n_features=conv_out_flat_size)
            self.actor_params += list(self.actor_fc_layers.parameters())

        elif isinstance(self.parameter.MODEL_PARAMETER, ParameterRecurrentLinearModel):
            input_n_features = self.observation_shape[0]
            self.actor_recurrent_layers = self.get_recurrent_layers(input_n_features=input_n_features)
            self.actor_params += list(self.actor_recurrent_layers.parameters())

            self.actor_fc_layers = self.get_linear_layers(self.parameter.MODEL_PARAMETER.HIDDEN_SIZE)
            self.actor_params += list(self.actor_fc_layers.parameters())

        elif isinstance(self.parameter.MODEL_PARAMETER, ParameterRecurrentConvolutionalModel):
            input_n_channels = self.observation_shape[0]
            self.actor_conv_layers = self.get_conv_layers(input_n_channels=input_n_channels)
            self.actor_params += list(self.actor_conv_layers.parameters())

            conv_out_flat_size = self._get_conv_out(self.actor_conv_layers, self.observation_shape)
            self.actor_fc_layers_1 = nn.Linear(conv_out_flat_size, self.parameter.MODEL_PARAMETER.HIDDEN_SIZE)
            self.actor_params += list(self.actor_fc_layers_1.parameters())

            self.actor_recurrent_layers = self.get_recurrent_layers(self.parameter.MODEL_PARAMETER.HIDDEN_SIZE)
            self.actor_params += list(self.actor_recurrent_layers.parameters())

            self.actor_fc_layers_2 = self.get_linear_layers(self.parameter.MODEL_PARAMETER.HIDDEN_SIZE)
            self.actor_params += list(self.actor_fc_layers_2.parameters())
        else:
            raise ValueError()

    def forward_actor(self, obs, save_hidden=False):
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float32, device=self.parameter.DEVICE)

        if isinstance(self.parameter.MODEL_PARAMETER, ParameterLinearModel):
            x = self.actor_fc_layers(obs)

        elif isinstance(self.parameter.MODEL_PARAMETER, ParameterConvolutionalModel):
            conv_out = self.actor_conv_layers(obs)
            conv_out = torch.flatten(conv_out, start_dim=1)
            x = self.actor_fc_layers(conv_out)

        elif isinstance(self.parameter.MODEL_PARAMETER, ParameterRecurrentLinearModel):
            rnn_in, h_0 = obs[0]
            if isinstance(rnn_in, np.ndarray):
                rnn_in = torch.tensor(rnn_in, dtype=torch.float32, device=self.parameter.DEVICE)
            if isinstance(h_0, np.ndarray):
                h_0 = torch.tensor(h_0, dtype=torch.float32, device=self.parameter.DEVICE)

            if rnn_in.ndim == 2:
                rnn_in = rnn_in.unsqueeze(1)

            rnn_out, h_n = self.actor_recurrent_layers(rnn_in, h_0)

            if save_hidden:
                self.recurrent_hidden = h_n.detach()

            rnn_out_flattened = torch.flatten(rnn_out, start_dim=1)

            x = self.actor_fc_layers(rnn_out_flattened)

        elif isinstance(self.parameter.MODEL_PARAMETER, ParameterRecurrentConvolutionalModel):
            x, h_0 = obs[0]
            if isinstance(x, np.ndarray):
                x = torch.tensor(x, dtype=torch.float32, device=self.parameter.DEVICE)
            if isinstance(h_0, np.ndarray):
                h_0 = torch.tensor(h_0, dtype=torch.float32, device=self.parameter.DEVICE)

            conv_out = self.actor_conv_layers(x)
            conv_out = torch.flatten(conv_out, start_dim=1)
            x = self.actor_fc_layers_1(conv_out)

            rnn_in = x
            if rnn_in.ndim == 2:
                rnn_in = rnn_in.unsqueeze(1)

            rnn_out, h_n = self.actor_recurrent_layers(rnn_in, h_0)
            self.recurrent_hidden = h_n.detach()  # save hidden
            rnn_out = torch.flatten(rnn_out, start_dim=1)
            x = self.actor_fc_layers_2(rnn_out)

        else:
            raise ValueError()

        return x

    @abstractmethod
    def pi(self, x):
        pass


class DiscretePolicyModel(PolicyModel):
    def __init__(self, observation_shape, n_out_actions, n_discrete_actions, parameter):
        super(DiscretePolicyModel, self).__init__(
            observation_shape, n_out_actions, n_discrete_actions, parameter
        )

        self.actor_fc_pi = nn.Linear(self.parameter.MODEL_PARAMETER.NEURONS_PER_FULLY_CONNECTED_LAYER[-1], self.n_discrete_actions)
        self.actor_params += list(self.actor_fc_pi.parameters())

    def pi(self, obs, save_hidden=False):
        x = self.forward_actor(obs, save_hidden=save_hidden)
        x = self.actor_fc_pi(x)
        action_prob = F.softmax(x, dim=-1)
        return action_prob


class ContinuousPolicyModel(PolicyModel):
    def __init__(self, observation_shape, n_out_actions, parameter=None):
        super(ContinuousPolicyModel, self).__init__(
            observation_shape=observation_shape, n_out_actions=n_out_actions, parameter=parameter
        )
        self.mu = nn.Sequential(
            nn.Linear(self.parameter.MODEL_PARAMETER.NEURONS_PER_FULLY_CONNECTED_LAYER[-1], self.n_out_actions),
            nn.Tanh()
        )
        self.actor_params += list(self.mu.parameters())

        self.var = nn.Sequential(
            nn.Linear(self.parameter.MODEL_PARAMETER.NEURONS_PER_FULLY_CONNECTED_LAYER[-1], self.n_out_actions),
            nn.Softplus()
        )
        self.actor_params += list(self.var.parameters())

        # if parameter.AGENT_TYPE == AgentType.SAC:
        #     self.logstd = nn.Sequential(
        #         nn.Linear(self.parameter.MODEL_PARAMETER.NEURONS_PER_FULLY_CONNECTED_LAYER[-1], self.n_out_actions),
        #         nn.Softplus()
        #     )
        #     self.actor_params += list(self.logstd.parameters())
        # else:
        #     logstds_param = nn.Parameter(torch.full((self.n_out_actions,), 0.1))
        #     self.register_parameter("logstds", logstds_param)
        #     self.actor_params.append(self.logstds)

    def pi(self, obs, save_hidden=False):
        x = self.forward_actor(obs, save_hidden=save_hidden)

        mu_v = self.mu(x)

        var_v = self.var(x)
        var_v = torch.clamp(var_v, 1e-2, 256)
#       std_v = torch.exp(F.softplus(self.logstds))

        # logstd_v = self.logstd(x)
        # std_v = torch.exp(logstd_v)

        # if self.parameter.AGENT_TYPE == AgentType.SAC:
        #     logstd_v = self.logstd(x)
        #     std_v = torch.exp(logstd_v)
        # else:
        #     std_v = F.softplus(self.logstds.exp())

        return mu_v, var_v


DiscreteActorModel = DiscretePolicyModel
ContinuousActorModel = ContinuousPolicyModel
