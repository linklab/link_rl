from typing import Tuple
import torch
from torch import nn

from link_rl.a_configuration.a_base_config.c_models.config_convolutional_models import Config2DConvolutionalModel, \
    Config1DConvolutionalModel
from link_rl.a_configuration.a_base_config.c_models.config_linear_models import ConfigLinearModel
from link_rl.a_configuration.a_base_config.c_models.config_recurrent_convolutional_models import \
    ConfigRecurrent2DConvolutionalModel, ConfigRecurrent1DConvolutionalModel
from link_rl.a_configuration.a_base_config.c_models.config_recurrent_linear_models import ConfigRecurrentLinearModel
from link_rl.c_models.a_models import Model
from link_rl.g_utils.types import ConvolutionType


class QNet(Model):
    # self.n_out_actions: 1
    # self.n_discrete_actions: 4 (for gridworld)
    def __init__(
        self,
        observation_shape: Tuple[int],
        n_out_actions: int,
        n_discrete_actions=None,
        config=None
    ):
        super(QNet, self).__init__(
            observation_shape=observation_shape,
            n_out_actions=n_out_actions,
            n_discrete_actions=n_discrete_actions,
            config=config
        )

        if isinstance(self.config.MODEL_PARAMETER, ConfigLinearModel):
            self.make_linear_model(
                observation_shape=observation_shape, activation=self.config.LAYER_ACTIVATION()
            )

        elif isinstance(self.config.MODEL_PARAMETER, Config1DConvolutionalModel):
            self.make_convolutional_model(
                observation_shape=observation_shape, activation=self.config.LAYER_ACTIVATION(),
                convolution_type=ConvolutionType.ONE_DIMENSION
            )

        elif isinstance(self.config.MODEL_PARAMETER, Config2DConvolutionalModel):
            self.make_convolutional_model(
                observation_shape=observation_shape, activation=self.config.LAYER_ACTIVATION(),
                convolution_type=ConvolutionType.TWO_DIMENSION
            )

        elif isinstance(self.config.MODEL_PARAMETER, ConfigRecurrentLinearModel):
            self.make_recurrent_linear_model(
                observation_shape=observation_shape, activation=self.config.LAYER_ACTIVATION()
            )

        elif isinstance(self.config.MODEL_PARAMETER, ConfigRecurrent1DConvolutionalModel):
            self.make_recurrent_convolutional_model(
                observation_shape=observation_shape, activation=self.config.LAYER_ACTIVATION(),
                convolution_type=ConvolutionType.ONE_DIMENSION
            )

        elif isinstance(self.config.MODEL_PARAMETER, ConfigRecurrent2DConvolutionalModel):
            self.make_recurrent_convolutional_model(
                observation_shape=observation_shape, activation=self.config.LAYER_ACTIVATION(),
                convolution_type=ConvolutionType.TWO_DIMENSION
            )

        else:
            raise ValueError()

        self.linear_last_layer = nn.Linear(
            self._get_forward_pre_out(observation_shape), self.n_discrete_actions
        )

        self.qnet_params_list = list(self.parameters())
        self.version = 0

    def q(self, obs, save_hidden=False):
        x = self._forward(obs, save_hidden=save_hidden)
        q_values = self.linear_last_layer(x)
        return q_values


class DuelingQNet(QNet):
    # self.n_out_actions: 1
    # self.n_discrete_actions: 4 (for gridworld)
    def __init__(
        self,
        observation_shape: Tuple[int],
        n_out_actions: int,
        n_discrete_actions=None,
        config=None
    ):
        super(DuelingQNet, self).__init__(
            observation_shape=observation_shape,
            n_out_actions=n_out_actions,
            n_discrete_actions=n_discrete_actions,
            config=config
        )

        self.linear_last_adv = nn.Linear(
            self.config.MODEL_PARAMETER.NEURONS_PER_FULLY_CONNECTED_LAYER[-1], self.n_discrete_actions
        )

        self.linear_last_val = nn.Linear(
            self.config.MODEL_PARAMETER.NEURONS_PER_FULLY_CONNECTED_LAYER[-1], 1
        )

        self.qnet_params_list = list(self.parameters())
        self.version = 0

    def q(self, obs, save_hidden=False):
        x = self._forward(obs, save_hidden=save_hidden)
        adv = self.linear_last_adv(x)
        val = self.linear_last_val(x)
        q_values = val + adv - torch.mean(adv, dim=-1, keepdim=True)

        return q_values