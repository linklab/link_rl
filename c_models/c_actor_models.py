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


class ActorModel(Model):
    def __init__(
            self,
            observation_shape: Tuple[int],
            n_out_actions: int,
            n_discrete_actions=None,
            config=None
    ):
        super(ActorModel, self).__init__(observation_shape, n_out_actions, n_discrete_actions, config)

        if isinstance(self.config.MODEL_PARAMETER, ConfigLinearModel):
            self.make_linear_model(observation_shape=observation_shape)

        elif isinstance(self.config.MODEL_PARAMETER, ConfigConvolutionalModel):
            self.make_convolutional_model(observation_shape=observation_shape)

        elif isinstance(self.config.MODEL_PARAMETER, ConfigRecurrentLinearModel):
            self.make_recurrent_linear_model(observation_shape=observation_shape)

        elif isinstance(self.config.MODEL_PARAMETER, ConfigRecurrentConvolutionalModel):
            self.make_recurrent_convolutional_model(observation_shape=observation_shape)

        else:
            raise ValueError()

    def forward_actor(self, obs, save_hidden=False):
        return self._forward(obs, save_hidden=save_hidden)

    @abstractmethod
    def pi(self, x):
        pass


class DiscreteActorModel(ActorModel):
    def __init__(self, observation_shape, n_out_actions, n_discrete_actions, config):
        super(DiscreteActorModel, self).__init__(
            observation_shape, n_out_actions, n_discrete_actions, config
        )

        self.actor_linear_pi = nn.Linear(
            self.config.MODEL_PARAMETER.NEURONS_PER_FULLY_CONNECTED_LAYER[-1], self.n_discrete_actions
        )
        self.actor_params_list = list(self.parameters())

    def pi(self, obs, save_hidden=False):
        x = self.forward_actor(obs, save_hidden=save_hidden)
        x = self.actor_linear_pi(x)
        action_prob = F.softmax(x, dim=-1)
        return action_prob


DiscretePolicyModel = DiscreteActorModel


class ContinuousDeterministicActorModel(ActorModel):
    def __init__(
            self, observation_shape: Tuple[int], n_out_actions: int, config=None
    ):
        super(ContinuousDeterministicActorModel, self).__init__(
            observation_shape=observation_shape, n_out_actions=n_out_actions, config=config
        )

        self.mu = nn.Sequential(
            nn.Linear(self.config.MODEL_PARAMETER.NEURONS_PER_FULLY_CONNECTED_LAYER[-1], self.n_out_actions),
            nn.Tanh()
        )
        self.actor_params_list = list(self.parameters())

    def pi(self, x, save_hidden=False):
        x = self.forward_actor(x, save_hidden=save_hidden)
        mu_v = self.mu(x)
        return mu_v


class ContinuousStochasticActorModel(ActorModel):
    def __init__(self, observation_shape, n_out_actions, config=None):
        super(ContinuousStochasticActorModel, self).__init__(
            observation_shape=observation_shape, n_out_actions=n_out_actions, config=config
        )
        self.mu = nn.Sequential(
            nn.Linear(self.config.MODEL_PARAMETER.NEURONS_PER_FULLY_CONNECTED_LAYER[-1], self.n_out_actions),
            nn.Tanh()
        )

        # We handle the log of the standard deviation as the torch parameter.
        # log_sigma = 0.1 <- starting value. it mean std = 1.105
        # log_sigma_param = nn.Parameter(torch.full((self.n_out_actions,), 0.1))
        # self.register_parameter("log_sigma", log_sigma_param)

        self.sigma = nn.Sequential(
            nn.Linear(self.config.MODEL_PARAMETER.NEURONS_PER_FULLY_CONNECTED_LAYER[-1], self.n_out_actions),
            nn.Softplus()
        )

        # self.var = nn.Sequential(
        #     nn.Linear(self.config.MODEL_PARAMETER.NEURONS_PER_FULLY_CONNECTED_LAYER[-1], self.n_out_actions),
        #     nn.Softplus()
        # )

        self.actor_params_list = list(self.parameters())

    def forward_sigma(self, obs):
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float32, device=self.config.DEVICE)

        s = self.representation_layers(obs)
        sigma_v = self.sigma(s)

        return sigma_v

    def pi(self, obs, save_hidden=False):
        x = self.forward_actor(obs, save_hidden=save_hidden)
        mu_v = self.mu(x)

        # We just have to exponentiate it to have the standard deviation
        # By doing so we ensure we don’t have negative values with numerical stability too.
        # The standard deviation can’t be negative (nor 0).
        # sigma_v = torch.clamp(self.log_sigma.exp(), 1e-3, 50)

        sigma_v = self.forward_sigma(obs)
        sigma_v = torch.clamp(sigma_v, 1e-3, 50)

        return mu_v, sigma_v
