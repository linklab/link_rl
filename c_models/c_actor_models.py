from abc import abstractmethod
from typing import Tuple
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from a_configuration.a_base_config.c_models.config_convolutional_models import Config2DConvolutionalModel, \
    Config1DConvolutionalModel
from a_configuration.a_base_config.c_models.config_linear_models import ConfigLinearModel
from a_configuration.a_base_config.c_models.config_recurrent_convolutional_models import \
    ConfigRecurrent2DConvolutionalModel, ConfigRecurrent1DConvolutionalModel
from a_configuration.a_base_config.c_models.config_recurrent_linear_models import ConfigRecurrentLinearModel
from c_models.a_models import Model
from g_utils.types import ConvolutionType


class ActorModel(Model):
    def __init__(
        self,
        observation_shape: Tuple[int],
        n_out_actions: int,
        n_discrete_actions=None,
        config=None
    ):
        super(ActorModel, self).__init__(
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

    def forward_actor(self, obs):
        return self._forward(obs)

    @abstractmethod
    def pi(self, x):
        raise NotImplementedError()


class DiscreteActorModel(ActorModel):
    def __init__(
        self,
        observation_shape: Tuple[int],
        n_out_actions: int,
        n_discrete_actions=None,
        config=None
    ):
        super(DiscreteActorModel, self).__init__(
            observation_shape=observation_shape,
            n_out_actions=n_out_actions,
            n_discrete_actions=n_discrete_actions,
            config=config
        )

        self.actor_linear_pi = nn.Linear(
            self._get_forward_pre_out(observation_shape), self.n_discrete_actions
        )
        self.actor_params_list = list(self.parameters())

    def pi(self, obs):
        x = self.forward_actor(obs)
        x = self.actor_linear_pi(x)
        action_prob = F.softmax(x, dim=-1)
        return action_prob


class ContinuousDeterministicActorModel(ActorModel):
    def __init__(
        self,
        observation_shape: Tuple[int],
        n_out_actions: int,
        config=None
    ):
        super(ContinuousDeterministicActorModel, self).__init__(
            observation_shape=observation_shape,
            n_out_actions=n_out_actions,
            config=config
        )

        self.mu = nn.Sequential(
            nn.Linear(
                self._get_forward_pre_out(observation_shape), self.n_out_actions
            ),
            nn.Tanh()
        )
        self.actor_params_list = list(self.parameters())

    def pi(self, x):
        x = self.forward_actor(x)
        mu_v = self.mu(x)
        return mu_v


class ContinuousStochasticActorModel(ActorModel):
    def __init__(
        self,
        observation_shape: Tuple[int],
        n_out_actions: int,
        config=None
    ):
        super(ContinuousStochasticActorModel, self).__init__(
            observation_shape=observation_shape,
            n_out_actions=n_out_actions,
            config=config
        )
        self.mu = nn.Sequential(
            nn.Linear(self._get_forward_pre_out(observation_shape), self.n_out_actions),
            nn.Tanh()
        )

        self.variance = nn.Sequential(
            nn.Linear(self.config.MODEL_PARAMETER.NEURONS_PER_REPRESENTATION_LAYER[-1], self.n_out_actions),
            nn.Softplus()
        )

        #self.variance = torch.full(size=(self.n_out_actions,), fill_value=1.0)

        # We handle the log of the standard deviation as the torch parameter.
        # log_sigma = 0.1 <- starting value. it mean std = 1.105
        # log_sigma_param = nn.Parameter(torch.full((self.n_out_actions,), 0.1))
        # self.register_parameter("log_sigma", log_sigma_param)

        # self.sigma = nn.Sequential(
        #     nn.Linear(self.config.MODEL_PARAMETER.NEURONS_PER_FULLY_CONNECTED_LAYER[-1], self.n_out_actions),
        #     nn.Softplus()
        # )

        self.actor_params_list = list(self.parameters())

    def forward_variance(self, obs):
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float32, device=self.config.DEVICE)

        if any([
            isinstance(self.config.MODEL_PARAMETER, ConfigLinearModel),
            isinstance(self.config.MODEL_PARAMETER, ConfigRecurrentLinearModel)
        ]):
            x = self.representation_layers(obs)

        elif any([
            isinstance(self.config.MODEL_PARAMETER, Config2DConvolutionalModel),
            isinstance(self.config.MODEL_PARAMETER, ConfigRecurrent2DConvolutionalModel)
        ]):
            x = self.convolutional_layers(obs)
            x = torch.flatten(x, start_dim=1)
            x = self.representation_layers(x)

        else:
            raise ValueError()

        return x

    def pi(self, obs):
        x = self.forward_actor(obs)
        mu_v = self.mu(x)

        # We just have to exponentiate it to have the standard deviation
        # By doing so we ensure we don’t have negative values with numerical stability too.
        # The standard deviation can’t be negative (nor 0).
        # sigma_v = torch.clamp(self.log_sigma.exp(), 1e-3, 50)
        # sigma_v = self.log_sigma(x).exp()

        x = self.forward_variance(obs)
        var_v = self.variance(x)
        var_v = torch.clamp(var_v, 1e-4, 50)

        return mu_v, var_v
