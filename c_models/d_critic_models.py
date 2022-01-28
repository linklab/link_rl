from abc import abstractmethod
from typing import Tuple
import numpy as np
import torch
from torch import nn

from a_configuration.a_base_config.c_models.convolutional_models import ConfigConvolutionalModel
from a_configuration.a_base_config.c_models.linear_models import ConfigLinearModel
from a_configuration.a_base_config.c_models.recurrent_convolutional_models import ConfigRecurrentConvolutionalModel
from a_configuration.a_base_config.c_models.recurrent_linear_models import ConfigRecurrentLinearModel
from c_models.a_models import Model


class CriticModel(Model):
    def _critic_forward(self, obs):
        pass


class ValueCriticModel(CriticModel):
    def __init__(
        self,
        observation_shape: Tuple[int],
        n_out_actions: int,
        n_discrete_actions=None,
        config=None
    ):
        super(ValueCriticModel, self).__init__(
            observation_shape=observation_shape,
            n_out_actions=n_out_actions,
            n_discrete_actions=n_discrete_actions,
            config=config
        )

        if any([
            isinstance(self.config.MODEL_PARAMETER, ConfigLinearModel),
            isinstance(self.config.MODEL_PARAMETER, ConfigRecurrentLinearModel)
        ]):
            self.make_linear_model(observation_shape=observation_shape)

        elif any([
            isinstance(self.config.MODEL_PARAMETER, ConfigConvolutionalModel),
            isinstance(self.config.MODEL_PARAMETER, ConfigRecurrentConvolutionalModel)
        ]):
            self.make_convolutional_model(observation_shape=observation_shape)

        else:
            raise ValueError()

        self.critic_linear_last_layer = nn.Linear(self.config.MODEL_PARAMETER.NEURONS_PER_FULLY_CONNECTED_LAYER[-1], 1)
        self.critic_linear_last_layer = nn.Linear(self.config.MODEL_PARAMETER.NEURONS_PER_FULLY_CONNECTED_LAYER[-1], 1)

        self.critic_params_list = list(self.parameters())

    def forward_critic(self, obs):
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
            obs, _ = obs[0]
            x = self.representation_layers(obs)
            x = self.linear_layers(x)

        elif isinstance(self.config.MODEL_PARAMETER, ConfigRecurrentConvolutionalModel):
            conv_out = self.convolutional_layers(obs)
            conv_out = torch.flatten(conv_out, start_dim=1)
            x = self.representation_layers(conv_out)
            x = self.linear_layers(x)

        else:
            raise ValueError()

        return x

    def v(self, obs):
        x = self.forward_critic(obs)
        value = self.critic_linear_last_layer(x)
        return value


class QCriticModel(CriticModel):
    def __init__(
            self,
            observation_shape: Tuple[int],
            n_out_actions: int,
            n_discrete_actions=None,
            config=None
    ):
        super(QCriticModel, self).__init__(
            observation_shape=observation_shape,
            n_out_actions=n_out_actions,
            n_discrete_actions=n_discrete_actions,
            config=config
        )

        if any([
            isinstance(self.config.MODEL_PARAMETER, ConfigLinearModel),
            isinstance(self.config.MODEL_PARAMETER, ConfigRecurrentLinearModel)
        ]):
            self.make_linear_model(
                observation_shape=observation_shape, n_out_actions=self.n_out_actions
            )

        elif any([
            isinstance(self.config.MODEL_PARAMETER, ConfigConvolutionalModel),
            isinstance(self.config.MODEL_PARAMETER, ConfigRecurrentConvolutionalModel)
        ]):
            self.make_convolutional_model(
                observation_shape=observation_shape, n_out_actions=self.n_out_actions
            )

        else:
            raise ValueError()

        self.critic_linear_last_layer = nn.Linear(self.config.MODEL_PARAMETER.NEURONS_PER_FULLY_CONNECTED_LAYER[-1], 1)
        
        self.critic_params_list = list(self.parameters())

    def forward_critic(self, obs, act):
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float32, device=self.config.DEVICE)
        if isinstance(act, np.ndarray):
            act = torch.tensor(act, dtype=torch.float32, device=self.config.DEVICE)

        if isinstance(self.config.MODEL_PARAMETER, ConfigLinearModel):
            x = self.representation_layers(obs)

            x = torch.cat([x, act], dim=-1)
            x = self.linear_layers(x)

        elif isinstance(self.config.MODEL_PARAMETER, ConfigConvolutionalModel):
            conv_out = self.convolutional_layers(obs)
            conv_out = torch.flatten(conv_out, start_dim=1)
            x = self.representation_layers(conv_out)

            x = torch.cat([x, act], dim=-1)
            x = self.linear_layers(x)

        elif isinstance(self.config.MODEL_PARAMETER, ConfigRecurrentLinearModel):
            obs, _ = obs[0]
            x = self.representation_layers(obs)

            x = torch.cat([x, act], dim=-1)
            x = self.linear_layers(x)

        elif isinstance(self.config.MODEL_PARAMETER, ConfigRecurrentConvolutionalModel):
            obs, _ = obs[0]
            conv_out = self.convolutional_layers(obs)
            conv_out = torch.flatten(conv_out, start_dim=1)
            x = self.representation_layers(conv_out)

            x = torch.cat([x, act], dim=-1)
            x = self.linear_layers(x)

        else:
            raise ValueError()

        return x

    def q(self, obs, act):
        x = self.forward_critic(obs, act)
        q_value = self.critic_linear_last_layer(x)
        return q_value
    
    
class DoubleQCriticModel(CriticModel):
    def __init__(
            self,
            observation_shape: Tuple[int],
            n_out_actions: int,
            n_discrete_actions=None,
            config=None
    ):
        super(DoubleQCriticModel, self).__init__(
            observation_shape=observation_shape,
            n_out_actions=n_out_actions,
            n_discrete_actions=n_discrete_actions,
            config=config
        )

        if any([
            isinstance(self.config.MODEL_PARAMETER, ConfigLinearModel),
            isinstance(self.config.MODEL_PARAMETER, ConfigRecurrentLinearModel)
        ]):
            input_n_features = observation_shape[0]
            self.representation_layers = self.get_representation_layers(input_n_features=input_n_features)

            input_n_features = self.config.MODEL_PARAMETER.NEURONS_PER_REPRESENTATION_LAYER[-1] + n_out_actions

            # q1
            self.q1_linear_layers = self.get_linear_layers(input_n_features=input_n_features)

            # q2
            self.q2_linear_layers = self.get_linear_layers(input_n_features=input_n_features)

        elif any([
            isinstance(self.config.MODEL_PARAMETER, ConfigConvolutionalModel),
            isinstance(self.config.MODEL_PARAMETER, ConfigRecurrentConvolutionalModel)
        ]):
            input_n_channels = self.observation_shape[0]
            self.convolutional_layers = self.get_convolutional_layers(input_n_channels=input_n_channels)
            conv_out_flat_size = self._get_conv_out(self.convolutional_layers, self.observation_shape)
            self.representation_layers = self.get_representation_layers(input_n_features=conv_out_flat_size)

            # q1
            input_n_features = self.config.MODEL_PARAMETER.NEURONS_PER_REPRESENTATION_LAYER[-1] + n_out_actions

            self.q1_linear_layers = self.get_linear_layers(input_n_features=input_n_features)

            # q2
            self.q2_linear_layers = self.get_linear_layers(input_n_features=input_n_features)

        else:
            raise ValueError()

        # q1
        self.q1_linear_last_layer = nn.Linear(self.config.MODEL_PARAMETER.NEURONS_PER_FULLY_CONNECTED_LAYER[-1], 1)

        # q2
        self.q2_linear_last_layer = nn.Linear(self.config.MODEL_PARAMETER.NEURONS_PER_FULLY_CONNECTED_LAYER[-1], 1)

        self.critic_params_list = list(self.parameters())

    def forward_critic(self, obs, act):
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float32, device=self.config.DEVICE)
        if isinstance(act, np.ndarray):
            act = torch.tensor(act, dtype=torch.float32, device=self.config.DEVICE)

        if isinstance(self.config.MODEL_PARAMETER, ConfigLinearModel):
            x = self.representation_layers(obs)

            x = torch.cat([x, act], dim=-1).float()
            q1_x = self.q1_linear_layers(x)
            q2_x = self.q2_linear_layers(x)

        elif isinstance(self.config.MODEL_PARAMETER, ConfigConvolutionalModel):
            conv_out = self.convolutional_layers(obs)
            conv_out = torch.flatten(conv_out, start_dim=1)
            x = self.representation_layers(conv_out)

            x = torch.cat([x, act], dim=-1)
            q1_x = self.q1_linear_layers(x)
            q2_x = self.q2_linear_layers(x)

        elif isinstance(self.config.MODEL_PARAMETER, ConfigRecurrentLinearModel):
            obs, _ = obs[0]
            x = self.representation_layers(obs)

            x = torch.cat([x, act], dim=-1)
            q1_x = self.q1_linear_layers(x)
            q2_x = self.q2_linear_layers(x)

        elif isinstance(self.config.MODEL_PARAMETER, ConfigRecurrentConvolutionalModel):
            obs, _ = obs[0]
            conv_out = self.convolutional_layers(obs)
            conv_out = torch.flatten(conv_out, start_dim=1)
            x = self.representation_layers(conv_out)

            x = torch.cat([x, act], dim=-1)
            q1_x = self.q1_linear_layers(x)
            q2_x = self.q2_linear_layers(x)

        else:
            raise ValueError()

        return q1_x, q2_x

    def q(self, obs, act):
        q1_x, q2_x = self.forward_critic(obs, act)
        q1_value = self.q1_linear_last_layer(q1_x)
        q2_value = self.q2_linear_last_layer(q2_x)
        return q1_value, q2_value
