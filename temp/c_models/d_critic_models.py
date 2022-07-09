from abc import abstractmethod
from typing import Tuple
import numpy as np
import torch
from torch import nn

from link_rl.a_configuration.a_base_config.c_models.config_convolutional_models import Config2DConvolutionalModel, \
    Config1DConvolutionalModel
from link_rl.a_configuration.a_base_config.c_models.config_linear_models import ConfigLinearModel
from link_rl.a_configuration.a_base_config.c_models.config_recurrent_convolutional_models import \
    ConfigRecurrent2DConvolutionalModel, ConfigRecurrent1DConvolutionalModel
from link_rl.a_configuration.a_base_config.c_models.config_recurrent_linear_models import ConfigRecurrentLinearModel
from link_rl.c_models.a_models import Model
from link_rl.h_utils.types import ConvolutionType


class CriticModel(Model):
    @abstractmethod
    def _forward(self, obs):
        raise NotImplementedError()

    @abstractmethod
    def _forward(self, obs, act):
        raise NotImplementedError()

    def forward_critic(self, obs):
        raise NotImplementedError()

    def _get_forward_pre_out_with_act(self, observation_shape, n_out_actions):
        obs = torch.zeros(1, *observation_shape)
        act = torch.zeros(1, n_out_actions)
        forward_pre_out = self._forward(obs, act)

        return int(np.prod(forward_pre_out.size()))


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
            self.make_linear_model(
                observation_shape=observation_shape, activation=self.config.VALUE_NETWORK_LAYER_ACTIVATION()
            )

        elif any([
            isinstance(self.config.MODEL_PARAMETER, Config1DConvolutionalModel),
            isinstance(self.config.MODEL_PARAMETER, ConfigRecurrent1DConvolutionalModel)
        ]):
            self.make_convolutional_model(
                observation_shape=observation_shape, activation=self.config.VALUE_NETWORK_LAYER_ACTIVATION(),
                convolution_type=ConvolutionType.ONE_DIMENSION
            )

        elif any([
            isinstance(self.config.MODEL_PARAMETER, Config2DConvolutionalModel),
            isinstance(self.config.MODEL_PARAMETER, ConfigRecurrent2DConvolutionalModel)
        ]):
            self.make_convolutional_model(
                observation_shape=observation_shape, activation=self.config.VALUE_NETWORK_LAYER_ACTIVATION(),
                convolution_type=ConvolutionType.TWO_DIMENSION
            )

        else:
            raise ValueError()

        self.critic_linear_last_layer = nn.Linear(
            self._get_forward_pre_out(observation_shape), 1
        )

        self.critic_params_list = list(self.parameters())

    def _forward(self, obs):
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float32, device=self.config.DEVICE)

        if isinstance(self.config.MODEL_PARAMETER, ConfigLinearModel):
            x = self.representation_layers(obs)
            x = self.linear_layers(x)

        elif isinstance(self.config.MODEL_PARAMETER, Config2DConvolutionalModel):
            encoder_out = self.convolutional_layers(obs)
            encoder_out = torch.flatten(encoder_out, start_dim=1)
            x = self.representation_layers(encoder_out)
            x = self.linear_layers(x)

        elif isinstance(self.config.MODEL_PARAMETER, ConfigRecurrentLinearModel):
            obs, _ = obs[0]
            x = self.representation_layers(obs)
            x = self.linear_layers(x)

        elif isinstance(self.config.MODEL_PARAMETER, ConfigRecurrent2DConvolutionalModel):
            encoder_out = self.convolutional_layers(obs)
            encoder_out = torch.flatten(encoder_out, start_dim=1)
            x = self.representation_layers(encoder_out)
            x = self.linear_layers(x)

        else:
            raise ValueError()

        return x

    def forward_critic(self, obs):
        return self._forward(obs)

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
                observation_shape=observation_shape, n_out_actions=self.n_out_actions,
                activation=self.config.VALUE_NETWORK_LAYER_ACTIVATION()
            )

        elif any([
            isinstance(self.config.MODEL_PARAMETER, Config2DConvolutionalModel),
            isinstance(self.config.MODEL_PARAMETER, ConfigRecurrent2DConvolutionalModel)
        ]):
            self.make_convolutional_model(
                observation_shape=observation_shape, n_out_actions=self.n_out_actions,
                activation=self.config.VALUE_NETWORK_LAYER_ACTIVATION()
            )

        else:
            raise ValueError()

        self.critic_linear_last_layer = nn.Linear(
            self.config.MODEL_PARAMETER.NEURONS_PER_FULLY_CONNECTED_LAYER[-1], 1
        )
        # self.critic_linear_last_layer = nn.Linear(
        #     self._get_forward_pre_out_with_act(observation_shape, n_out_actions), 1
        # )
        
        self.critic_params_list = list(self.parameters())

    def _forward(self, obs, act):
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float32, device=self.config.DEVICE)
        if isinstance(act, np.ndarray):
            act = torch.tensor(act, dtype=torch.float32, device=self.config.DEVICE)

        if isinstance(self.config.MODEL_PARAMETER, ConfigLinearModel):
            x = self.representation_layers(obs)

            x = torch.cat([x, act], dim=-1).float()
            x = self.linear_layers(x)

        elif isinstance(self.config.MODEL_PARAMETER, Config2DConvolutionalModel):
            encoder_out = self.convolutional_layers(obs)

            encoder_out = torch.flatten(encoder_out, start_dim=1)
            x = self.representation_layers(encoder_out)

            x = torch.cat([x, act], dim=-1).float()
            x = self.linear_layers(x)

        elif isinstance(self.config.MODEL_PARAMETER, ConfigRecurrentLinearModel):
            obs, _ = obs[0]
            x = self.representation_layers(obs)

            x = torch.cat([x, act], dim=-1).float()
            x = self.linear_layers(x)

        elif isinstance(self.config.MODEL_PARAMETER, ConfigRecurrent2DConvolutionalModel):
            obs, _ = obs[0]
            encoder_out = self.convolutional_layers(obs)

            encoder_out = torch.flatten(encoder_out, start_dim=1)
            x = self.representation_layers(encoder_out)

            x = torch.cat([x, act], dim=-1).float()
            x = self.linear_layers(x)

        else:
            raise ValueError()

        return x

    def forward_critic(self, obs, act):
        return self._forward(obs, act)

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

            if self.config.MODEL_PARAMETER.NEURONS_PER_REPRESENTATION_LAYER:
                input_n_features = self.config.MODEL_PARAMETER.NEURONS_PER_REPRESENTATION_LAYER[-1] + n_out_actions
            else:
                input_n_features = input_n_features + n_out_actions

            # q1
            self.q1_linear_layers = self.get_linear_layers(input_n_features=input_n_features)

            # q2
            self.q2_linear_layers = self.get_linear_layers(input_n_features=input_n_features)

        elif any([
            isinstance(self.config.MODEL_PARAMETER, Config2DConvolutionalModel),
            isinstance(self.config.MODEL_PARAMETER, ConfigRecurrent2DConvolutionalModel)
        ]):
            input_n_channels = self.observation_shape[0]
            self.convolutional_layers = self.get_convolutional_layers(input_n_channels=input_n_channels)
            encoder_out_flat_size = self._get_encoder_out(self.convolutional_layers, self.observation_shape)
            self.representation_layers = self.get_representation_layers(input_n_features=encoder_out_flat_size)

            if self.config.MODEL_PARAMETER.NEURONS_PER_REPRESENTATION_LAYER:
                input_n_features = self.config.MODEL_PARAMETER.NEURONS_PER_REPRESENTATION_LAYER[-1] + n_out_actions
            else:
                input_n_features = encoder_out_flat_size + n_out_actions

            # q1
            self.q1_linear_layers = self.get_linear_layers(input_n_features=input_n_features)

            # q2
            self.q2_linear_layers = self.get_linear_layers(input_n_features=input_n_features)

        else:
            raise ValueError()

        # q1
        self.q1_linear_last_layer = nn.Linear(
            self.config.MODEL_PARAMETER.NEURONS_PER_FULLY_CONNECTED_LAYER[-1], 1
        )
        # self.q1_linear_last_layer = nn.Linear(
        #     self._get_forward_pre_out_with_act(observation_shape, n_out_actions), 1
        # )

        # q2
        self.q2_linear_last_layer = nn.Linear(
            self.config.MODEL_PARAMETER.NEURONS_PER_FULLY_CONNECTED_LAYER[-1], 1
        )
        # self.q2_linear_last_layer = nn.Linear(
        #     self._get_forward_pre_out_with_act(observation_shape, n_out_actions), 1
        # )

        self.critic_params_list = list(self.parameters())

    def _forward(self, obs, act):
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float32, device=self.config.DEVICE)
        if isinstance(act, np.ndarray):
            act = torch.tensor(act, dtype=torch.float32, device=self.config.DEVICE)

        if isinstance(self.config.MODEL_PARAMETER, ConfigLinearModel):
            x = self.representation_layers(obs)

            x = torch.cat([x, act], dim=-1).float()
            q1_x = self.q1_linear_layers(x)
            q2_x = self.q2_linear_layers(x)

        elif isinstance(self.config.MODEL_PARAMETER, Config2DConvolutionalModel):
            encoder_out = self.convolutional_layers(obs)
            encoder_out = torch.flatten(encoder_out, start_dim=1)
            x = self.representation_layers(encoder_out)

            x = torch.cat([x, act], dim=-1).float()
            q1_x = self.q1_linear_layers(x)
            q2_x = self.q2_linear_layers(x)

        elif isinstance(self.config.MODEL_PARAMETER, ConfigRecurrentLinearModel):
            obs, _ = obs[0]
            x = self.representation_layers(obs)

            x = torch.cat([x, act], dim=-1).float()
            q1_x = self.q1_linear_layers(x)
            q2_x = self.q2_linear_layers(x)

        elif isinstance(self.config.MODEL_PARAMETER, ConfigRecurrent2DConvolutionalModel):
            obs, _ = obs[0]
            encoder_out = self.convolutional_layers(obs)
            encoder_out = torch.flatten(encoder_out, start_dim=1)
            x = self.representation_layers(encoder_out)

            x = torch.cat([x, act], dim=-1).float()
            q1_x = self.q1_linear_layers(x)
            q2_x = self.q2_linear_layers(x)

        else:
            raise ValueError()

        return q1_x, q2_x

    def forward_critic(self, obs, act):
        return self._forward(obs, act)

    def q(self, obs, act):
        q1_x, q2_x = self.forward_critic(obs, act)
        q1_value = self.q1_linear_last_layer(q1_x)
        q2_value = self.q2_linear_last_layer(q2_x)

        return q1_value, q2_value
