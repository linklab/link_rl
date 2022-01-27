from typing import Tuple
import torch
from torch import nn
import numpy as np
from a_configuration.a_base_config.c_models.convolutional_models import ConfigConvolutionalModel
from a_configuration.a_base_config.c_models.linear_models import ConfigLinearModel
from a_configuration.a_base_config.c_models.recurrent_convolutional_models import ConfigRecurrentConvolutionalModel
from a_configuration.a_base_config.c_models.recurrent_linear_models import ConfigRecurrentLinearModel
from c_models.a_models import Model
from c_models.c_policy_models import DiscreteActorModel, ContinuousActorModel
from torch.distributions import Normal, TanhTransform, TransformedDistribution
from e_main.config import config


class SacCriticModel(Model):
    def __init__(
            self, observation_shape: Tuple[int], n_out_actions: int, n_discrete_actions=None, config=None
    ):
        super(SacCriticModel, self).__init__(observation_shape, n_out_actions, n_discrete_actions, config)

        self.critic_params = []

        if any([
            isinstance(self.config.MODEL_PARAMETER, ConfigLinearModel),
            isinstance(self.config.MODEL_PARAMETER, ConfigRecurrentLinearModel)
        ]):
            input_n_features = self.observation_shape[0] + self.n_out_actions

            # q1
            self.q1_fc_layers = self.get_linear_layers(input_n_features=input_n_features)
            self.critic_params += list(self.q1_fc_layers.parameters())

            # q2
            self.q2_fc_layers = self.get_linear_layers(input_n_features=input_n_features)
            self.critic_params += list(self.q2_fc_layers.parameters())

        elif any([
            isinstance(self.config.MODEL_PARAMETER, ConfigConvolutionalModel),
            isinstance(self.config.MODEL_PARAMETER, ConfigRecurrentConvolutionalModel)
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

        # elif isinstance(self.config.MODEL_PARAMETER, ConfigRecurrentLinearModel):
        #     input_n_features = self.observation_shape[0] + self.n_out_actions
        #
        #     self.recurrent_layers = self.get_recurrent_layers(input_n_features=input_n_features)
        #     self.critic_params += list(self.recurrent_layers.parameters())
        #
        #     # q1
        #     self.q1_fc_layers = self.get_linear_layers(self.config.MODEL_PARAMETER.HIDDEN_SIZE)
        #     self.critic_params += list(self.q1_fc_layers.parameters())
        #
        #     # q2
        #     self.q2_fc_layers = self.get_linear_layers(self.config.MODEL_PARAMETER.HIDDEN_SIZE)
        #     self.critic_params += list(self.q2_fc_layers.parameters())
        #
        # elif isinstance(self.config.MODEL_PARAMETER, ConfigRecurrentConvolutionalModel):
        #     input_n_channels = self.observation_shape[0]
        #     self.conv_layers = self.get_conv_layers(input_n_channels=input_n_channels)
        #     self.critic_params += list(self.conv_layers.parameters())
        #
        #     conv_out_flat_size = self._get_conv_out(self.conv_layers, self.observation_shape)
        #     input_n_features = conv_out_flat_size + self.n_out_actions
        #
        #     self.fc_layers_1 = nn.Linear(
        #         in_features=input_n_features, out_features=self.config.MODEL_PARAMETER.HIDDEN_SIZE
        #     )
        #     self.critic_params += list(self.fc_layers_1.parameters())
        #
        #     self.recurrent_layers = self.get_recurrent_layers(self.config.MODEL_PARAMETER.HIDDEN_SIZE)
        #     self.critic_params += list(self.recurrent_layers.parameters())
        #
        #     # q1
        #     self.q1_fc_layers_2 = self.get_linear_layers(self.config.MODEL_PARAMETER.HIDDEN_SIZE)
        #     self.critic_params += list(self.q1_fc_layers_2.parameters())
        #
        #     # q2
        #     self.q2_fc_layers_2 = self.get_linear_layers(self.config.MODEL_PARAMETER.HIDDEN_SIZE)
        #     self.critic_params += list(self.q2_fc_layers_2.parameters())

        else:
            raise ValueError()

        # q1
        self.q1_fc_last_layer = nn.Linear(self.config.MODEL_PARAMETER.NEURONS_PER_FULLY_CONNECTED_LAYER[-1], 1)
        self.critic_params += list(self.q1_fc_last_layer.parameters())

        # q2
        self.q2_fc_last_layer = nn.Linear(self.config.MODEL_PARAMETER.NEURONS_PER_FULLY_CONNECTED_LAYER[-1], 1)
        self.critic_params += list(self.q2_fc_last_layer.parameters())

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
            # rnn_in, hidden_in = obs[0]
            # if isinstance(rnn_in, np.ndarray):
            #     rnn_in = torch.tensor(rnn_in, dtype=torch.float32, device=self.config.DEVICE)
            # if isinstance(hidden_in, np.ndarray):
            #     hidden_in = torch.tensor(hidden_in, dtype=torch.float32, device=self.config.DEVICE)
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

        elif isinstance(self.config.MODEL_PARAMETER, ConfigRecurrentConvolutionalModel):
            obs, _ = obs[0]
            conv_out = self.q1_conv_layers(obs)
            conv_out = torch.flatten(conv_out, start_dim=1)

            q1_x = self.q1_fc_layers(torch.cat([conv_out, act], dim=-1))
            q2_x = self.q2_fc_layers(torch.cat([conv_out, act], dim=-1))
            # x, hidden_in = obs[0]
            # if isinstance(x, np.ndarray):
            #     x = torch.tensor(x, dtype=torch.float32, device=self.config.DEVICE)
            # if isinstance(hidden_in, np.ndarray):
            #     hidden_in = torch.tensor(hidden_in, dtype=torch.float32, device=self.config.DEVICE)
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
            self, observation_shape: Tuple[int], n_out_actions: int, n_discrete_actions=None, config=None,
            is_target_model=False
    ):
        self.config = config

        if is_target_model:
            self.actor_model = None
        else:
            self.actor_model = DiscreteActorModel(
                observation_shape=observation_shape, n_out_actions=n_out_actions, n_discrete_actions=n_discrete_actions,
                config=self.config
            ).to(self.config.DEVICE)

        self.critic_model = SacCriticModel(
            observation_shape=observation_shape, n_out_actions=n_out_actions, n_discrete_actions=n_discrete_actions,
            config=self.config
        ).to(self.config.DEVICE)


class ContinuousSacModel:
    def __init__(
            self, observation_shape: Tuple[int], n_out_actions: int, config=None, is_target_model=False
    ):
        self.config = config

        if is_target_model:
            self.actor_model = None
        else:
            self.actor_model = ContinuousActorModel(
                observation_shape=observation_shape, n_out_actions=n_out_actions, config=config
            ).to(self.config.DEVICE)

        self.critic_model = SacCriticModel(
            observation_shape=observation_shape, n_out_actions=n_out_actions, n_discrete_actions=None,
            config=self.config
        ) .to(self.config.DEVICE)

    def re_parameterization_trick_sample(self, obs):
        mu_v, sigma_v = self.actor_model.pi(obs)

        dist = Normal(loc=mu_v, scale=sigma_v)
        dist = TransformedDistribution(base_distribution=dist, transforms=TanhTransform(cache_size=1))

        action_v = dist.rsample()  # for reparameterization trick (mean + std * N(0,1))

        log_probs = dist.log_prob(action_v).sum(dim=-1, keepdim=True)
        # action_v.shape: [128, 1]
        # log_prob.shape: [128, 1]
        entropy = 0.5 * (torch.log(2.0 * np.pi * sigma_v ** 2) + 1.0).sum(dim=-1)

        return action_v, log_probs, entropy

    # https://github.com/pranz24/pytorch-soft-actor-critic/blob/398595e0d9dca98b7db78c7f2f939c969431871a/model.py#L64
    # def re_parameterization_trick_sample(self, obs):
    #     mu_v, sigma_v = self.actor_model.pi(obs)
    #     normal = Normal(loc=mu_v, scale=sigma_v)
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
    a = SacCriticModel((4,), 1, 4, config=config)