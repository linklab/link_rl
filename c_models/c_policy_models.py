from abc import abstractmethod
from typing import Tuple
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from a_configuration.a_base_config.c_models.convolutional_models import ConfigConvolutionalModel
from a_configuration.a_base_config.c_models.linear_models import ConfigLinearModel
from a_configuration.a_base_config.c_models.recurrent_convolutional_models import ConfigRecurrentConvolutionalModel
from a_configuration.a_base_config.c_models.recurrent_linear_models import ConfigRecurrentLinearModel
from c_models.a_models import Model


class PolicyModel(Model):
    def __init__(
            self, observation_shape: Tuple[int], n_out_actions: int, n_discrete_actions=None, config=None
    ):
        super(PolicyModel, self).__init__(observation_shape, n_out_actions, n_discrete_actions, config)

        self.actor_params = []
        if isinstance(self.config.MODEL_PARAMETER, ConfigLinearModel):
            input_n_features = self.observation_shape[0]
            self.actor_fc_layers = self.get_linear_layers(input_n_features=input_n_features)
            self.actor_params += list(self.actor_fc_layers.parameters())

        elif isinstance(self.config.MODEL_PARAMETER, ConfigConvolutionalModel):
            input_n_channels = self.observation_shape[0]
            self.actor_conv_layers = self.get_conv_layers(input_n_channels=input_n_channels)
            self.actor_params += list(self.actor_conv_layers.parameters())

            conv_out_flat_size = self._get_conv_out(self.actor_conv_layers, self.observation_shape)
            self.actor_fc_layers = self.get_linear_layers(input_n_features=conv_out_flat_size)
            self.actor_params += list(self.actor_fc_layers.parameters())

        elif isinstance(self.config.MODEL_PARAMETER, ConfigRecurrentLinearModel):
            input_n_features = self.observation_shape[0]
            self.actor_recurrent_layers = self.get_recurrent_layers(input_n_features=input_n_features)
            self.actor_params += list(self.actor_recurrent_layers.parameters())

            self.actor_fc_layers = self.get_linear_layers(self.config.MODEL_PARAMETER.HIDDEN_SIZE)
            self.actor_params += list(self.actor_fc_layers.parameters())

        elif isinstance(self.config.MODEL_PARAMETER, ConfigRecurrentConvolutionalModel):
            input_n_channels = self.observation_shape[0]
            self.actor_conv_layers = self.get_conv_layers(input_n_channels=input_n_channels)
            self.actor_params += list(self.actor_conv_layers.parameters())

            conv_out_flat_size = self._get_conv_out(self.actor_conv_layers, self.observation_shape)
            self.actor_fc_layers_1 = nn.Linear(conv_out_flat_size, self.config.MODEL_PARAMETER.HIDDEN_SIZE)
            self.actor_params += list(self.actor_fc_layers_1.parameters())

            self.actor_recurrent_layers = self.get_recurrent_layers(self.config.MODEL_PARAMETER.HIDDEN_SIZE)
            self.actor_params += list(self.actor_recurrent_layers.parameters())

            self.actor_fc_layers_2 = self.get_linear_layers(self.config.MODEL_PARAMETER.HIDDEN_SIZE)
            self.actor_params += list(self.actor_fc_layers_2.parameters())
        else:
            raise ValueError()

    def forward_actor(self, obs, save_hidden=False):
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float32, device=self.config.DEVICE)

        if isinstance(self.config.MODEL_PARAMETER, ConfigLinearModel):
            x = self.actor_fc_layers(obs)

        elif isinstance(self.config.MODEL_PARAMETER, ConfigConvolutionalModel):
            conv_out = self.actor_conv_layers(obs)
            conv_out = torch.flatten(conv_out, start_dim=1)
            x = self.actor_fc_layers(conv_out)

        elif isinstance(self.config.MODEL_PARAMETER, ConfigRecurrentLinearModel):
            rnn_in, h_0 = obs[0]
            if isinstance(rnn_in, np.ndarray):
                rnn_in = torch.tensor(rnn_in, dtype=torch.float32, device=self.config.DEVICE)
            if isinstance(h_0, np.ndarray):
                h_0 = torch.tensor(h_0, dtype=torch.float32, device=self.config.DEVICE)

            if rnn_in.ndim == 2:
                rnn_in = rnn_in.unsqueeze(1)

            rnn_out, h_n = self.actor_recurrent_layers(rnn_in, h_0)

            if save_hidden:
                self.recurrent_hidden = h_n.detach()

            rnn_out_flattened = torch.flatten(rnn_out, start_dim=1)

            x = self.actor_fc_layers(rnn_out_flattened)

        elif isinstance(self.config.MODEL_PARAMETER, ConfigRecurrentConvolutionalModel):
            x, h_0 = obs[0]
            if isinstance(x, np.ndarray):
                x = torch.tensor(x, dtype=torch.float32, device=self.config.DEVICE)
            if isinstance(h_0, np.ndarray):
                h_0 = torch.tensor(h_0, dtype=torch.float32, device=self.config.DEVICE)

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
    def __init__(self, observation_shape, n_out_actions, n_discrete_actions, config):
        super(DiscretePolicyModel, self).__init__(
            observation_shape, n_out_actions, n_discrete_actions, config
        )

        self.actor_fc_pi = nn.Linear(self.config.MODEL_PARAMETER.NEURONS_PER_FULLY_CONNECTED_LAYER[-1], self.n_discrete_actions)
        self.actor_params += list(self.actor_fc_pi.parameters())

    def pi(self, obs, save_hidden=False):
        x = self.forward_actor(obs, save_hidden=save_hidden)
        x = self.actor_fc_pi(x)
        action_prob = F.softmax(x, dim=-1)
        return action_prob


class ContinuousPolicyModel(PolicyModel):
    def __init__(self, observation_shape, n_out_actions, config=None):
        super(ContinuousPolicyModel, self).__init__(
            observation_shape=observation_shape, n_out_actions=n_out_actions, config=config
        )
        self.mu = nn.Sequential(
            nn.Linear(self.config.MODEL_PARAMETER.NEURONS_PER_FULLY_CONNECTED_LAYER[-1], self.n_out_actions),
            nn.Tanh()
        )
        self.actor_params += list(self.mu.parameters())

        # We handle the log of the standard deviation as the torch parameter.
        # log_sigma = 0.1 <- starting value. it mean std = 1.105
        log_sigma_param = nn.Parameter(torch.full((self.n_out_actions,), 0.1))
        self.register_parameter("log_sigma", log_sigma_param)
        self.actor_params.append(self.log_sigma)

        # self.sigma = nn.Sequential(
        #     nn.Linear(self.config.MODEL_PARAMETER.NEURONS_PER_FULLY_CONNECTED_LAYER[-1], self.n_out_actions),
        #     nn.Softplus()
        # )
        # self.actor_params += list(self.sigma.parameters())

        # self.var = nn.Sequential(
        #     nn.Linear(self.config.MODEL_PARAMETER.NEURONS_PER_FULLY_CONNECTED_LAYER[-1], self.n_out_actions),
        #     nn.Softplus()
        # )
        # self.actor_params += list(self.var.parameters())

    def pi(self, obs, save_hidden=False):
        x = self.forward_actor(obs, save_hidden=save_hidden)
        mu_v = self.mu(x)

        # We just have to exponentiate it to have the standard deviation
        # By doing so we ensure we don’t have negative values with numerical stability too.
        # The standard deviation can’t be negative (nor 0).
        sigma_v = torch.clamp(self.log_sigma.exp(), 1e-3, 50)
        return mu_v, sigma_v


DiscreteActorModel = DiscretePolicyModel
ContinuousActorModel = ContinuousPolicyModel
