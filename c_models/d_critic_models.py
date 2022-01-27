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
    def __init__(
            self, observation_shape: Tuple[int], n_out_actions: int, n_discrete_actions=None, config=None
    ):
        super(CriticModel, self).__init__(observation_shape, n_out_actions, n_discrete_actions, config)

        ############################
        # CRITIC MODEL_TYPE: BEGIN #
        ############################
        if any([
            isinstance(self.config.MODEL_PARAMETER, ConfigLinearModel),
            isinstance(self.config.MODEL_PARAMETER, ConfigRecurrentLinearModel)
        ]):
            input_n_features = self.observation_shape[0]
            self.critic_fc_layers = self.get_linear_layers(input_n_features=input_n_features)

        elif any([
            isinstance(self.config.MODEL_PARAMETER, ConfigConvolutionalModel),
            isinstance(self.config.MODEL_PARAMETER, ConfigRecurrentConvolutionalModel)
        ]):
            input_n_channels = self.observation_shape[0]
            self.critic_conv_layers = self.get_conv_layers(input_n_channels=input_n_channels)
            conv_out_flat_size = self._get_conv_out(self.critic_conv_layers, observation_shape)
            self.critic_fc_layers = self.get_linear_layers(input_n_features=conv_out_flat_size)

        else:
            raise ValueError()

        self.critic_fc_last_layer = nn.Linear(self.config.MODEL_PARAMETER.NEURONS_PER_FULLY_CONNECTED_LAYER[-1], 1)

        self.critic_params_list = list(self.parameters())
        ##########################
        # CRITIC MODEL_TYPE: END #
        ##########################

    def forward_critic(self, obs):
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float32, device=self.config.DEVICE)

        if isinstance(self.config.MODEL_PARAMETER, ConfigLinearModel):
            x = self.critic_fc_layers(obs)

        elif isinstance(self.config.MODEL_PARAMETER, ConfigConvolutionalModel):
            conv_out = self.critic_conv_layers(obs)
            conv_out = torch.flatten(conv_out, start_dim=1)
            x = self.critic_fc_layers(conv_out)

        elif isinstance(self.config.MODEL_PARAMETER, ConfigRecurrentLinearModel):
            obs, _ = obs[0]
            x = self.critic_fc_layers(obs)

        elif isinstance(self.config.MODEL_PARAMETER, ConfigRecurrentConvolutionalModel):
            obs, _ = obs[0]
            conv_out = self.critic_conv_layers(obs)
            conv_out = torch.flatten(conv_out, start_dim=1)
            x = self.critic_fc_layers(conv_out)

        else:
            raise ValueError()

        return x

    def v(self, obs):
        x = self.forward_critic(obs)
        value = self.critic_fc_last_layer(x)
        return value


class QCriticModel(Model):
    def __init__(
            self, observation_shape: Tuple[int], n_out_actions: int, n_discrete_actions=None, config=None
    ):
        super(QCriticModel, self).__init__(observation_shape, n_out_actions, n_discrete_actions, config)
        ############################
        # CRITIC MODEL_TYPE: BEGIN #
        ############################
        if any([
            isinstance(self.config.MODEL_PARAMETER, ConfigLinearModel),
            isinstance(self.config.MODEL_PARAMETER, ConfigRecurrentLinearModel)
        ]):
            input_n_features = self.observation_shape[0] + self.n_out_actions
            self.critic_fc_layers = self.get_linear_layers(input_n_features=input_n_features)

        elif any([
            isinstance(self.config.MODEL_PARAMETER, ConfigConvolutionalModel),
            isinstance(self.config.MODEL_PARAMETER, ConfigRecurrentConvolutionalModel)
        ]):
            input_n_channels = self.observation_shape[0]
            self.critic_conv_layers = self.get_conv_layers(input_n_channels=input_n_channels)

            conv_out_flat_size = self._get_conv_out(self.critic_conv_layers, self.observation_shape)
            input_n_features = conv_out_flat_size + self.n_out_actions
            self.critic_fc_layers = self.get_linear_layers(input_n_features=input_n_features)

        else:
            raise ValueError()

        self.critic_fc_last_layer = nn.Linear(self.config.MODEL_PARAMETER.NEURONS_PER_FULLY_CONNECTED_LAYER[-1], 1)
        
        self.critic_params_list = list(self.parameters())
        ##########################
        # CRITIC MODEL_TYPE: END #
        ##########################

    def forward_critic(self, obs, act):
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float32, device=self.config.DEVICE)

        if isinstance(act, np.ndarray):
            act = torch.tensor(act, dtype=torch.float32, device=self.config.DEVICE)

        if isinstance(self.config.MODEL_PARAMETER, ConfigLinearModel):
            x = self.critic_fc_layers(torch.cat([obs, act], dim=-1))

        elif isinstance(self.config.MODEL_PARAMETER, ConfigConvolutionalModel):
            conv_out = self.critic_conv_layers(obs)
            conv_out = torch.flatten(conv_out, start_dim=1)
            x = self.critic_fc_layers(torch.cat([conv_out, act], dim=-1))

        elif isinstance(self.config.MODEL_PARAMETER, ConfigRecurrentLinearModel):
            obs, _ = obs[0]
            x = self.critic_fc_layers(torch.cat([obs, act], dim=-1))

        elif isinstance(self.config.MODEL_PARAMETER, ConfigRecurrentConvolutionalModel):
            obs, _ = obs[0]
            conv_out = self.critic_conv_layers(obs)
            conv_out = torch.flatten(conv_out, start_dim=1)
            x = self.critic_fc_layers(torch.cat([conv_out, act], dim=-1))
            
        else:
            raise ValueError()

        return x

    def q(self, obs, act):
        x = self.forward_critic(obs, act)
        q_value = self.critic_fc_last_layer(x)
        return q_value
    
    
class DoubleQCriticModel(Model):
    def __init__(
            self, observation_shape: Tuple[int], n_out_actions: int, n_discrete_actions=None, config=None
    ):
        super(DoubleQCriticModel, self).__init__(observation_shape, n_out_actions, n_discrete_actions, config)

        if any([
            isinstance(self.config.MODEL_PARAMETER, ConfigLinearModel),
            isinstance(self.config.MODEL_PARAMETER, ConfigRecurrentLinearModel)
        ]):
            input_n_features = self.observation_shape[0] + self.n_out_actions

            # q1
            self.q1_fc_layers = self.get_linear_layers(input_n_features=input_n_features)

            # q2
            self.q2_fc_layers = self.get_linear_layers(input_n_features=input_n_features)

        elif any([
            isinstance(self.config.MODEL_PARAMETER, ConfigConvolutionalModel),
            isinstance(self.config.MODEL_PARAMETER, ConfigRecurrentConvolutionalModel)
        ]):
            input_n_channels = self.observation_shape[0]
            self.conv_layers = self.get_conv_layers(input_n_channels=input_n_channels)

            conv_out_flat_size = self._get_conv_out(self.conv_layers, self.observation_shape)
            input_n_features = conv_out_flat_size + self.n_out_actions

            # q1
            self.q1_fc_layers = self.get_linear_layers(input_n_features=input_n_features)

            # q2
            self.q2_fc_layers = self.get_linear_layers(input_n_features=input_n_features)
        else:
            raise ValueError()

        # q1
        self.q1_fc_last_layer = nn.Linear(self.config.MODEL_PARAMETER.NEURONS_PER_FULLY_CONNECTED_LAYER[-1], 1)

        # q2
        self.q2_fc_last_layer = nn.Linear(self.config.MODEL_PARAMETER.NEURONS_PER_FULLY_CONNECTED_LAYER[-1], 1)

        self.critic_params_list = list(self.parameters())

    def forward_critic(self, obs, act):
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float32, device=self.config.DEVICE)
        if isinstance(act, np.ndarray):
            act = torch.tensor(act, dtype=torch.float32, device=self.config.DEVICE)

        if isinstance(self.config.MODEL_PARAMETER, ConfigLinearModel):
            q1_x = self.q1_fc_layers(torch.cat([obs, act], dim=-1))
            q2_x = self.q2_fc_layers(torch.cat([obs, act], dim=-1))

        elif isinstance(self.config.MODEL_PARAMETER, ConfigConvolutionalModel):
            conv_out = self.q1_conv_layers(obs)
            conv_out = torch.flatten(conv_out, start_dim=1)

            q1_x = self.q1_fc_layers(torch.cat([conv_out, act], dim=-1))
            q2_x = self.q2_fc_layers(torch.cat([conv_out, act], dim=-1))

        elif isinstance(self.config.MODEL_PARAMETER, ConfigRecurrentLinearModel):
            obs, _ = obs[0]
            q1_x = self.q1_fc_layers(torch.cat([obs, act], dim=-1))
            q2_x = self.q2_fc_layers(torch.cat([obs, act], dim=-1))

        elif isinstance(self.config.MODEL_PARAMETER, ConfigRecurrentConvolutionalModel):
            obs, _ = obs[0]
            conv_out = self.q1_conv_layers(obs)
            conv_out = torch.flatten(conv_out, start_dim=1)

            q1_x = self.q1_fc_layers(torch.cat([conv_out, act], dim=-1))
            q2_x = self.q2_fc_layers(torch.cat([conv_out, act], dim=-1))

        else:
            raise ValueError()

        return q1_x, q2_x

    def q(self, obs, act):
        q1_x, q2_x = self.forward_critic(obs, act)
        q1_value = self.q1_fc_last_layer(q1_x)
        q2_value = self.q2_fc_last_layer(q2_x)
        return q1_value, q2_value
