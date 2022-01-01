from abc import abstractmethod
from typing import Tuple
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from a_configuration.b_base.c_models.convolutional_models import ParameterConvolutionalModel
from a_configuration.b_base.c_models.linear_models import ParameterLinearModel
from a_configuration.b_base.c_models.recurrent_models import ParameterRecurrentModel
from c_models.a_models import Model
from g_utils.types import AgentType


class PolicyModel(Model):
    def __init__(
            self, observation_shape: Tuple[int], n_out_actions: int, n_discrete_actions=None,
            device=torch.device("cpu"), parameter=None
    ):
        super(PolicyModel, self).__init__(observation_shape, n_out_actions, n_discrete_actions, device, parameter)
        self.actor_params = []
        if isinstance(self.parameter.MODEL, ParameterLinearModel):
            input_n_features = self.observation_shape[0]
            self.actor_fc_layers = self.get_linear_layers(input_n_features=input_n_features)
            self.actor_params += list(self.actor_fc_layers.parameters())
        elif isinstance(self.parameter.MODEL, ParameterConvolutionalModel):
            input_n_channels = self.observation_shape[0]
            self.actor_conv_layers = self.get_conv_layers(input_n_channels=input_n_channels)
            self.actor_params += list(self.actor_conv_layers.parameters())
            conv_out_flat_size = self._get_conv_out(self.actor_conv_layers, observation_shape)
            self.actor_fc_layers = self.get_linear_layers(input_n_features=conv_out_flat_size)
            self.actor_params += list(self.actor_fc_layers.parameters())
        elif isinstance(self.parameter.MODEL, ParameterRecurrentModel):
            pass
        else:
            raise ValueError()

    def forward_actor(self, x):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32, device=self.device)

        if isinstance(self.parameter.MODEL, ParameterLinearModel):
            x = self.actor_fc_layers(x)
        elif isinstance(self.parameter.MODEL, ParameterConvolutionalModel):
            conv_out = self.actor_conv_layers(x)
            conv_out = torch.flatten(conv_out, start_dim=1)
            x = self.actor_fc_layers(conv_out)
        else:
            raise ValueError()

        return x

    @abstractmethod
    def pi(self, x):
        pass


class DiscretePolicyModel(PolicyModel):
    def __init__(
            self, observation_shape: Tuple[int], n_out_actions: int, n_discrete_actions=None,
            device=torch.device("cpu"), parameter=None
    ):
        super(DiscretePolicyModel, self).__init__(
            observation_shape, n_out_actions, n_discrete_actions, device, parameter
        )

        self.actor_fc_pi = nn.Linear(self.parameter.MODEL.NEURONS_PER_FULLY_CONNECTED_LAYER[-1], self.n_discrete_actions)
        self.actor_params += list(self.actor_fc_pi.parameters())

    def pi(self, obs):
        x = self.forward_actor(obs)
        x = self.actor_fc_pi(x)
        action_prob = F.softmax(x, dim=-1)
        return action_prob


class ContinuousPolicyModel(PolicyModel):
    def __init__(
            self, observation_shape: Tuple[int], n_out_actions: int, device=torch.device("cpu"), parameter=None
    ):
        super(ContinuousPolicyModel, self).__init__(
            observation_shape=observation_shape, n_out_actions=n_out_actions, device=device, parameter=parameter
        )
        self.mu = nn.Sequential(
            nn.Linear(self.parameter.MODEL.NEURONS_PER_FULLY_CONNECTED_LAYER[-1], self.n_out_actions),
            nn.Tanh()
        )
        self.actor_params += list(self.mu.parameters())

        self.var = nn.Sequential(
            nn.Linear(self.parameter.MODEL.NEURONS_PER_FULLY_CONNECTED_LAYER[-1], self.n_out_actions),
            nn.Softplus()
        )
        self.actor_params += list(self.var.parameters())

        # if parameter.AGENT_TYPE == AgentType.SAC:
        #     self.logstd = nn.Sequential(
        #         nn.Linear(self.parameter.MODEL.NEURONS_PER_FULLY_CONNECTED_LAYER[-1], self.n_out_actions),
        #         nn.Softplus()
        #     )
        #     self.actor_params += list(self.logstd.parameters())
        # else:
        #     logstds_param = nn.Parameter(torch.full((self.n_out_actions,), 0.1))
        #     self.register_parameter("logstds", logstds_param)
        #     self.actor_params.append(self.logstds)

    def pi(self, obs):
        x = self.forward_actor(obs)

        mu_v = self.mu(x)

        var_v = self.var(x)
#        std_v = torch.exp(F.softplus(self.logstds))

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
