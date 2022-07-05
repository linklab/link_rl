import enum

import torch
from torch import nn
from typing import Tuple, final

from link_rl.c_models_v2.a_model import DoubleModel, model_registry
from link_rl.g_utils.types import EncoderType


class SAC_MODEL(enum.Enum):
    ContinuousSacSharedModel = "ContinuousSacSharedModel"
    ContinuousSacModel = "ContinuousSacModel"
    ContinuousSacSharedEncoderModel = "ContinuousSacSharedEncoderModel"
    ContinuousSacEncoderModel = "ContinuousSacEncoderModel"


@model_registry.add
class ContinuousSacSharedModel(DoubleModel):
    class ActorModel(nn.Module):
        def __init__(self, shared_net, actor_net, actor_mu_net, actor_var_net):
            super().__init__()
            self.shared_net = shared_net
            self.actor_net = actor_net
            self.actor_mu_net = actor_mu_net
            self.actor_var_net = actor_var_net

        def forward(self, obs):
            x = self.shared_net(obs)
            x = self.actor_net(x)
            mu = self.actor_mu_net(x)
            var = self.actor_var_net(x)
            return mu, var

    class CriticModel(nn.Module):
        def __init__(self, shared_net, critic_net, q1_critic_net, q2_critic_net):
            super().__init__()
            self.shared_net = shared_net
            self.critic_net = critic_net
            self.q1_critic_net = q1_critic_net
            self.q2_critic_net = q2_critic_net

        def forward(self, obs, action):
            x = self.shared_net(obs)
            x = torch.cat([x, action], dim=-1).float()
            x = self.critic_net(x)
            q1 = self.q1_critic_net(x)
            q2 = self.q2_critic_net(x)

            return q1, q2

    def __init__(
        self,
        observation_shape: Tuple[int, ...],
        n_out_actions: int,
        n_discrete_actions=None
    ):
        super(ContinuousSacSharedModel, self).__init__(
            observation_shape,
            n_out_actions,
            n_discrete_actions
        )

    @final
    def _create_model(self) -> Tuple[nn.Module, nn.Module]:
        shared_net = nn.Sequential(
            nn.Linear(self._n_input, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(),
        )

        actor_net = nn.Sequential(
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(),
        )

        actor_mu_net = nn.Sequential(
            nn.Linear(128, self._n_out_actions),
            nn.Tanh()
        )

        actor_var_net = nn.Sequential(
            nn.Linear(128, self._n_out_actions),
            nn.Softplus()
        )

        critic_net = nn.Sequential(
            nn.Linear(128 + self._n_out_actions, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(),
        )
        q1_critic_net = nn.Linear(128, 1)
        q2_critic_net = nn.Linear(128, 1)

        actor_model = ContinuousSacSharedModel.ActorModel(shared_net, actor_net, actor_mu_net, actor_var_net)
        critic_model = ContinuousSacSharedModel.CriticModel(shared_net, critic_net, q1_critic_net, q2_critic_net)
        return actor_model, critic_model


@model_registry.add
class ContinuousSacModel(DoubleModel):
    class ActorModel(nn.Module):
        def __init__(self, actor_net, actor_mu_net, actor_var_net):
            super().__init__()
            self.actor_net = actor_net
            self.actor_mu_net = actor_mu_net
            self.actor_var_net = actor_var_net

        def forward(self, obs):
            x = self.actor_net(obs)
            mu = self.actor_mu_net(x)
            var = self.actor_var_net(x)
            return mu, var

    class CriticModel(nn.Module):
        def __init__(self, critic_net, representation_net, q1_critic_net, q2_critic_net):
            super().__init__()
            self.critic_net = critic_net
            self.representation_net = representation_net
            self.q1_critic_net = q1_critic_net
            self.q2_critic_net = q2_critic_net

        def forward(self, obs, action):
            x = self.representation_net(obs)
            x = torch.cat([x, action], dim=-1).float()
            x = self.critic_net(x)
            q1 = self.q1_critic_net(x)
            q2 = self.q2_critic_net(x)

            return q1, q2

    def __init__(
        self,
        observation_shape: Tuple[int, ...],
        n_out_actions: int,
        n_discrete_actions=None
    ):
        super(ContinuousSacModel, self).__init__(
            observation_shape,
            n_out_actions,
            n_discrete_actions
        )

    @final
    def _create_model(self) -> Tuple[nn.Module, nn.Module]:
        actor_net = nn.Sequential(
            nn.Linear(self._n_input, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(),
        )

        actor_mu_net = nn.Sequential(
            nn.Linear(128, self._n_out_actions),
            nn.Tanh()
        )

        actor_var_net = nn.Sequential(
            nn.Linear(128, self._n_out_actions),
            nn.Softplus()
        )

        representation_net = nn.Sequential(
            nn.Linear(self._n_input, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU()
        )

        critic_net = nn.Sequential(
            nn.Linear(128 + self._n_out_actions, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(),
        )
        q1_critic_net = nn.Linear(128, 1)
        q2_critic_net = nn.Linear(128, 1)

        actor_model = ContinuousSacModel.ActorModel(actor_net, actor_mu_net, actor_var_net)
        critic_model = ContinuousSacModel.CriticModel(critic_net, representation_net, q1_critic_net, q2_critic_net)
        return actor_model, critic_model


@model_registry.add
class ContinuousSacSharedEncoderModel(DoubleModel):
    class ActorModel(nn.Module):
        def __init__(self, encoder_net, shared_net, actor_net, actor_mu_net, actor_var_net):
            super().__init__()
            self.encoder_net = encoder_net
            self.shared_net = shared_net
            self.actor_net = actor_net
            self.actor_mu_net = actor_mu_net
            self.actor_var_net = actor_var_net

        def forward(self, obs):
            x = self.encoder_net(obs)
            x = self.shared_net(x)
            x = self.actor_net(x)
            mu = self.actor_mu_net(x)
            var = self.actor_var_net(x)
            return mu, var

    class CriticModel(nn.Module):
        def __init__(self, encoder_net, shared_net, critic_net, q1_critic_net, q2_critic_net):
            super().__init__()
            self.encoder_net = encoder_net
            self.shared_net = shared_net
            self.critic_net = critic_net
            self.q1_critic_net = q1_critic_net
            self.q2_critic_net = q2_critic_net

        def forward(self, obs, action):
            x = self.encoder_net(obs)
            x = self.shared_net(x)
            x = torch.cat([x, action], dim=-1).float()
            x = self.critic_net(x)
            q1 = self.q1_critic_net(x)
            q2 = self.q2_critic_net(x)

            return q1, q2

    def __init__(
        self,
        observation_shape: Tuple[int, ...],
        n_out_actions: int,
        n_discrete_actions=None,
        encoder_type=EncoderType.TWO_CONVOLUTION
    ):
        super(ContinuousSacSharedEncoderModel, self).__init__(
            observation_shape,
            n_out_actions,
            n_discrete_actions
        )
        self.encoder_type = encoder_type

    @final
    def _create_model(self) -> Tuple[nn.Module, nn.Module]:
        if self.encoder_type == EncoderType.TWO_CONVOLUTION:
            encoder_net = nn.Sequential(
                nn.Conv2d(in_channels=self._n_input, out_channels=16, kernel_size=(4, 4), stride=(2, 2)),
                nn.BatchNorm2d(16),
                nn.LeakyReLU(),
                nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(4, 4), stride=(2, 2)),
                nn.BatchNorm2d(32),
                nn.LeakyReLU(),
                nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(4, 4), stride=(2, 2)),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(),
            )
        else:
            raise ValueError()

        encoder_out = self._get_conv_out(encoder_net, self._observation_shape)

        shared_net = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(encoder_out, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(),
        )

        actor_net = nn.Sequential(
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(),
        )

        actor_mu_net = nn.Sequential(
            nn.Linear(128, self._n_out_actions),
            nn.Tanh()
        )

        actor_var_net = nn.Sequential(
            nn.Linear(128, self._n_out_actions),
            nn.Softplus()
        )

        critic_net = nn.Sequential(
            nn.Linear(128 + self._n_out_actions, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(),
        )
        q1_critic_net = nn.Linear(128, 1)
        q2_critic_net = nn.Linear(128, 1)

        actor_model = ContinuousSacSharedEncoderModel.ActorModel(
            encoder_net, shared_net, actor_net, actor_mu_net, actor_var_net
        )
        critic_model = ContinuousSacSharedEncoderModel.CriticModel(
            encoder_net, shared_net, critic_net, q1_critic_net, q2_critic_net
        )

        return actor_model, critic_model


@model_registry.add
class ContinuousSacEncoderModel(DoubleModel):
    class ActorModel(nn.Module):
        def __init__(self, encoder_net, actor_net, actor_mu_net, actor_var_net):
            super().__init__()
            self.encoder_net = encoder_net
            self.actor_net = actor_net
            self.actor_mu_net = actor_mu_net
            self.actor_var_net = actor_var_net

        def forward(self, obs):
            x = self.encoder_net(obs)
            x = x.flatten(start_dim=1)
            x = self.actor_net(x)
            mu = self.actor_mu_net(x)
            var = self.actor_var_net(x)
            return mu, var

    class CriticModel(nn.Module):
        def __init__(self, encoder_net, critic_net, q1_critic_net, q2_critic_net):
            super().__init__()
            self.encoder_net = encoder_net
            self.critic_net = critic_net
            self.q1_critic_net = q1_critic_net
            self.q2_critic_net = q2_critic_net

        def forward(self, obs, action):
            x = self.encoder_net(obs)
            x = x.flatten(start_dim=1)
            x = torch.cat([x, action], dim=-1).float()
            x = self.critic_net(x)
            q1 = self.q1_critic_net(x)
            q2 = self.q2_critic_net(x)

            return q1, q2

    def __init__(
        self,
        observation_shape: Tuple[int, ...],
        n_out_actions: int,
        n_discrete_actions=None,
        encoder_type=EncoderType.TWO_CONVOLUTION
    ):
        super(ContinuousSacEncoderModel, self).__init__(
            observation_shape,
            n_out_actions,
            n_discrete_actions
        )
        self.encoder_type = encoder_type

    @final
    def _create_model(self) -> Tuple[nn.Module, nn.Module]:
        if self.encoder_type == EncoderType.TWO_CONVOLUTION:
            actor_encoder_net = nn.Sequential(
                nn.Conv2d(in_channels=self._n_input, out_channels=16, kernel_size=(4, 4), stride=(2, 2)),
                nn.BatchNorm2d(16),
                nn.LeakyReLU(),
                nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(4, 4), stride=(2, 2)),
                nn.BatchNorm2d(32),
                nn.LeakyReLU(),
                nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(4, 4), stride=(2, 2)),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(),
            )
            critic_encoder_net = nn.Sequential(
                nn.Conv2d(in_channels=self._n_input, out_channels=16, kernel_size=(4, 4), stride=(2, 2)),
                nn.BatchNorm2d(16),
                nn.LeakyReLU(),
                nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(4, 4), stride=(2, 2)),
                nn.BatchNorm2d(32),
                nn.LeakyReLU(),
                nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(4, 4), stride=(2, 2)),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(),
            )
        else:
            raise ValueError()

        encoder_out = self._get_conv_out(actor_encoder_net, self._observation_shape)

        actor_net = nn.Sequential(
            nn.Linear(encoder_out, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(),
        )

        actor_mu_net = nn.Sequential(
            nn.Linear(128, self._n_out_actions),
            nn.Tanh()
        )

        actor_var_net = nn.Sequential(
            nn.Linear(128, self._n_out_actions),
            nn.Softplus()
        )

        critic_net = nn.Sequential(
            nn.Linear(encoder_out + self._n_out_actions, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(),
        )

        q1_critic_net = nn.Linear(128, 1)
        q2_critic_net = nn.Linear(128, 1)

        actor_model = ContinuousSacEncoderModel.ActorModel(
            actor_encoder_net, actor_net, actor_mu_net, actor_var_net
        )
        critic_model = ContinuousSacEncoderModel.CriticModel(
            critic_encoder_net, critic_net, q1_critic_net, q2_critic_net
        )

        return actor_model, critic_model
