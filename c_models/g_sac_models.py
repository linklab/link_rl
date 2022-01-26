from typing import Tuple
import torch
from torch import nn
import numpy as np
from a_configuration.b_base.c_models.convolutional_models import ParameterConvolutionalModel
from a_configuration.b_base.c_models.linear_models import ParameterLinearModel
from a_configuration.b_base.c_models.recurrent_convolutional_models import ParameterRecurrentConvolutionalModel
from a_configuration.b_base.c_models.recurrent_linear_models import ParameterRecurrentLinearModel
from c_models.a_models import Model
from c_models.c_policy_models import DiscreteActorModel, ContinuousActorModel
from torch.distributions import Normal, TanhTransform, TransformedDistribution
from e_main.parameter import parameter


class SacCriticModel(Model):
    def __init__(
            self, observation_shape: Tuple[int], n_out_actions: int, n_discrete_actions=None, parameter=None
    ):
        super(SacCriticModel, self).__init__(observation_shape, n_out_actions, n_discrete_actions, parameter)

        self.critic_params = []

        if any([
            isinstance(self.parameter.MODEL_PARAMETER, ParameterLinearModel),
            isinstance(self.parameter.MODEL_PARAMETER, ParameterRecurrentLinearModel)
        ]):
            input_n_features = self.observation_shape[0] + self.n_out_actions

            # q1
            self.q1_fc_layers = self.get_linear_layers(input_n_features=input_n_features)
            self.critic_params += list(self.q1_fc_layers.parameters())

            # q2
            self.q2_fc_layers = self.get_linear_layers(input_n_features=input_n_features)
            self.critic_params += list(self.q2_fc_layers.parameters())

        elif any([
            isinstance(self.parameter.MODEL_PARAMETER, ParameterConvolutionalModel),
            isinstance(self.parameter.MODEL_PARAMETER, ParameterRecurrentConvolutionalModel)
        ]):
            input_n_channels = self.observation_shape[0]
            self.conv_layers = self.get_conv_layers(input_n_channels=input_n_channels)
            self.critic_params += list(self.conv_layers.parameters())

            conv_out_flat_size = self._get_conv_out(self.conv_layers, self.observation_shape)
            input_n_features = conv_out_flat_size + self.n_out_actions

            # q1
            self.q1_fc_layers = self.get_linear_layers(input_n_features=input_n_features)
            self.critic_params += list(self.q1_fc_layers.parameters())

            # q2
            self.q2_fc_layers = self.get_linear_layers(input_n_features=input_n_features)
            self.critic_params += list(self.q2_fc_layers.parameters())

        # elif isinstance(self.parameter.MODEL_PARAMETER, ParameterRecurrentLinearModel):
        #     input_n_features = self.observation_shape[0] + self.n_out_actions
        #
        #     self.recurrent_layers = self.get_recurrent_layers(input_n_features=input_n_features)
        #     self.critic_params += list(self.recurrent_layers.parameters())
        #
        #     # q1
        #     self.q1_fc_layers = self.get_linear_layers(self.parameter.MODEL_PARAMETER.HIDDEN_SIZE)
        #     self.critic_params += list(self.q1_fc_layers.parameters())
        #
        #     # q2
        #     self.q2_fc_layers = self.get_linear_layers(self.parameter.MODEL_PARAMETER.HIDDEN_SIZE)
        #     self.critic_params += list(self.q2_fc_layers.parameters())
        #
        # elif isinstance(self.parameter.MODEL_PARAMETER, ParameterRecurrentConvolutionalModel):
        #     input_n_channels = self.observation_shape[0]
        #     self.conv_layers = self.get_conv_layers(input_n_channels=input_n_channels)
        #     self.critic_params += list(self.conv_layers.parameters())
        #
        #     conv_out_flat_size = self._get_conv_out(self.conv_layers, self.observation_shape)
        #     input_n_features = conv_out_flat_size + self.n_out_actions
        #
        #     self.fc_layers_1 = nn.Linear(
        #         in_features=input_n_features, out_features=self.parameter.MODEL_PARAMETER.HIDDEN_SIZE
        #     )
        #     self.critic_params += list(self.fc_layers_1.parameters())
        #
        #     self.recurrent_layers = self.get_recurrent_layers(self.parameter.MODEL_PARAMETER.HIDDEN_SIZE)
        #     self.critic_params += list(self.recurrent_layers.parameters())
        #
        #     # q1
        #     self.q1_fc_layers_2 = self.get_linear_layers(self.parameter.MODEL_PARAMETER.HIDDEN_SIZE)
        #     self.critic_params += list(self.q1_fc_layers_2.parameters())
        #
        #     # q2
        #     self.q2_fc_layers_2 = self.get_linear_layers(self.parameter.MODEL_PARAMETER.HIDDEN_SIZE)
        #     self.critic_params += list(self.q2_fc_layers_2.parameters())

        else:
            raise ValueError()

        # q1
        self.q1_fc_last_layer = nn.Linear(self.parameter.MODEL_PARAMETER.NEURONS_PER_FULLY_CONNECTED_LAYER[-1], 1)
        self.critic_params += list(self.q1_fc_last_layer.parameters())

        # q2
        self.q2_fc_last_layer = nn.Linear(self.parameter.MODEL_PARAMETER.NEURONS_PER_FULLY_CONNECTED_LAYER[-1], 1)
        self.critic_params += list(self.q2_fc_last_layer.parameters())

    def forward_critic(self, obs, act):
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float32, device=self.parameter.DEVICE)
        if isinstance(act, np.ndarray):
            act = torch.tensor(act, dtype=torch.float32, device=self.parameter.DEVICE)

        if isinstance(self.parameter.MODEL_PARAMETER, ParameterLinearModel):
            q1_x = self.q1_fc_layers(torch.cat([obs, act], dim=-1))
            q2_x = self.q2_fc_layers(torch.cat([obs, act], dim=-1))

        elif isinstance(self.parameter.MODEL_PARAMETER, ParameterConvolutionalModel):
            conv_out = self.q1_conv_layers(obs)
            conv_out = torch.flatten(conv_out, start_dim=1)

            q1_x = self.q1_fc_layers(torch.cat([conv_out, act], dim=-1))
            q2_x = self.q2_fc_layers(torch.cat([conv_out, act], dim=-1))

        elif isinstance(self.parameter.MODEL_PARAMETER, ParameterRecurrentLinearModel):
            obs, _ = obs[0]
            q1_x = self.q1_fc_layers(torch.cat([obs, act], dim=-1))
            q2_x = self.q2_fc_layers(torch.cat([obs, act], dim=-1))
            # rnn_in, hidden_in = obs[0]
            # if isinstance(rnn_in, np.ndarray):
            #     rnn_in = torch.tensor(rnn_in, dtype=torch.float32, device=self.parameter.DEVICE)
            # if isinstance(hidden_in, np.ndarray):
            #     hidden_in = torch.tensor(hidden_in, dtype=torch.float32, device=self.parameter.DEVICE)
            #
            # if act.ndim == 2:
            #     act = act.unsqueeze(1)
            #
            # if rnn_in.ndim == 2:
            #     rnn_in = rnn_in.unsqueeze(1)
            #
            # rnn_out, hidden_out = self.recurrent_layers(torch.cat([rnn_in, act], dim=-1), hidden_in)
            # self.recurrent_hidden = hidden_out.detach()  # save hidden
            # rnn_out_flattened = torch.flatten(rnn_out, start_dim=1)
            #
            # q1_x = self.q1_fc_layers(rnn_out_flattened)
            # q2_x = self.q2_fc_layers(rnn_out_flattened)

        elif isinstance(self.parameter.MODEL_PARAMETER, ParameterRecurrentConvolutionalModel):
            obs, _ = obs[0]
            conv_out = self.q1_conv_layers(obs)
            conv_out = torch.flatten(conv_out, start_dim=1)

            q1_x = self.q1_fc_layers(torch.cat([conv_out, act], dim=-1))
            q2_x = self.q2_fc_layers(torch.cat([conv_out, act], dim=-1))
            # x, hidden_in = obs[0]
            # if isinstance(x, np.ndarray):
            #     x = torch.tensor(x, dtype=torch.float32, device=self.parameter.DEVICE)
            # if isinstance(hidden_in, np.ndarray):
            #     hidden_in = torch.tensor(hidden_in, dtype=torch.float32, device=self.parameter.DEVICE)
            #
            # conv_out = self.critic_conv_layers(x)
            # conv_out = torch.flatten(conv_out, start_dim=1)
            # x = self.critic_fc_layers_1(conv_out)
            #
            # rnn_in = x
            # if act.ndim == 2:
            #     act = act.unsqueeze(1)
            #
            # if rnn_in.ndim == 2:
            #     rnn_in = rnn_in.unsqueeze(1)
            #
            # rnn_out, hidden_out = self.recurrent_layers(torch.cat([rnn_in, act], dim=-1), hidden_in)
            # self.recurrent_hidden = hidden_out.detach()  # save hidden
            # rnn_out_flattened = torch.flatten(rnn_out, start_dim=1)
            #
            # q1_x = self.q1_fc_layers_2(rnn_out_flattened)
            # q2_x = self.q2_fc_layers_2(rnn_out_flattened)

        else:
            raise ValueError()

        return q1_x, q2_x

    def q(self, obs, act):
        q1_x, q2_x = self.forward_critic(obs, act)
        q1_value = self.q1_fc_last_layer(q1_x)
        q2_value = self.q2_fc_last_layer(q2_x)
        return q1_value, q2_value


class DiscreteSacModel:
    def __init__(
            self, observation_shape: Tuple[int], n_out_actions: int, n_discrete_actions=None, parameter=None,
            is_target_model=False
    ):
        self.parameter = parameter

        if is_target_model:
            self.actor_model = None
        else:
            self.actor_model = DiscreteActorModel(
                observation_shape=observation_shape, n_out_actions=n_out_actions, n_discrete_actions=n_discrete_actions,
                parameter=self.parameter
            ).to(self.parameter.DEVICE)

        self.critic_model = SacCriticModel(
            observation_shape=observation_shape, n_out_actions=n_out_actions, n_discrete_actions=n_discrete_actions,
            parameter=self.parameter
        ).to(self.parameter.DEVICE)


class ContinuousSacModel:
    def __init__(
            self, observation_shape: Tuple[int], n_out_actions: int, parameter=None, is_target_model=False
    ):
        self.parameter = parameter

        if is_target_model:
            self.actor_model = None
        else:
            self.actor_model = ContinuousActorModel(
                observation_shape=observation_shape, n_out_actions=n_out_actions, parameter=parameter
            ).to(self.parameter.DEVICE)

        self.critic_model = SacCriticModel(
            observation_shape=observation_shape, n_out_actions=n_out_actions, n_discrete_actions=None,
            parameter=self.parameter
        ) .to(self.parameter.DEVICE)

    def re_parameterization_trick_sample(self, obs):
        mu_v, var_v = self.actor_model.pi(obs)

        dist = Normal(loc=mu_v, scale=torch.sqrt(var_v))
        dist = TransformedDistribution(base_distribution=dist, transforms=TanhTransform(cache_size=1))

        action_v = dist.rsample()  # for reparameterization trick (mean + std * N(0,1))

        log_probs = dist.log_prob(action_v).sum(dim=-1, keepdim=True)
        # action_v.shape: [128, 1]
        # log_prob.shape: [128, 1]
        entropy = 0.5 * (torch.log(2.0 * np.pi * var_v) + 1.0).sum(dim=-1)

        return action_v, log_probs, entropy

    # https://github.com/pranz24/pytorch-soft-actor-critic/blob/398595e0d9dca98b7db78c7f2f939c969431871a/model.py#L64
    # def re_parameterization_trick_sample(self, obs):
    #     mu_v, var_v = self.actor_model.pi(obs)
    #     normal = Normal(loc=mu_v, scale=torch.sqrt(var_v))
    #     action_v = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
    #     action_v = torch.tanh(action_v)
    #
    #     log_prob = normal.log_prob(action_v)
    #     # Enforcing Action Bound
    #     epsilon = 1e-06
    #     log_prob = log_prob - torch.log(1.0 - action_v.pow(2) + epsilon)
    #     log_prob = log_prob.sum(dim=-1, keepdim=True)
    #     return action_v, log_prob


if __name__ == "__main__":
    a = SacCriticModel((4,), 1, 4, parameter=parameter)