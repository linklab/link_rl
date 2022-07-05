import enum

from torch import nn
from typing import Tuple, final

from link_rl.c_models_v2.a_model import DoubleModel, model_registry
from link_rl.g_utils.types import EncoderType


class BASIC_ACTOR_CRITIC_MODEL(enum.Enum):
    DiscreteBasicActorCriticSharedModel = "DiscreteBasicActorCriticSharedModel"
    DiscreteBasicActorCriticEncoderSharedModel = "DiscreteBasicActorCriticEncoderSharedModel"
    ContinuousBasicActorCriticSharedModel = "ContinuousBasicActorCriticSharedModel"
    ContinuousBasicActorCriticEncoderSharedModel = "ContinuousBasicActorCriticEncoderSharedModel"


@model_registry.add
class DiscreteBasicActorCriticSharedModel(DoubleModel):
    def __init__(
        self,
        observation_shape: Tuple[int, ...],
        n_out_actions: int,
        n_discrete_actions=None
    ):
        super(DiscreteBasicActorCriticSharedModel, self).__init__(
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
            nn.Linear(128, self._n_discrete_actions),
            nn.Softmax(dim=-1)
        )

        critic_net = nn.Sequential(
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(),
            nn.Linear(128, 1)
        )

        actor_model = nn.Sequential(
            shared_net, actor_net
        )
        critic_model = nn.Sequential(
            shared_net, critic_net
        )
        return actor_model, critic_model


@model_registry.add
class DiscreteBasicActorCriticEncoderSharedModel(DoubleModel):
    def __init__(
        self,
        observation_shape: Tuple[int, ...],
        n_out_actions: int,
        n_discrete_actions=None,
        encoder_type=EncoderType.TWO_CONVOLUTION
    ):
        super(DiscreteBasicActorCriticEncoderSharedModel, self).__init__(
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
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1)),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(),
            )
        else:
            raise ValueError()

        encoder_out = self._get_conv_out(conv_layers=encoder_net, shape=self._observation_shape)

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
            nn.Linear(128, self._n_discrete_actions),
            nn.Softmax(dim=-1)
        )

        critic_net = nn.Sequential(
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(),
            nn.Linear(128, 1)
        )

        actor_model = nn.Sequential(
            encoder_net, shared_net, actor_net
        )
        critic_model = nn.Sequential(
            encoder_net, shared_net, critic_net
        )
        return actor_model, critic_model


@model_registry.add
class ContinuousBasicActorCriticSharedModel(DoubleModel):
    class ActorModel(nn.Module):
        def __init__(self, shared_net, actor_net, actor_mu_net, actor_var_net):
            super().__init__()
            self.share_net = shared_net
            self.actor_net = actor_net
            self.actor_mu_net = actor_mu_net
            self.actor_var_net = actor_var_net

        def forward(self, obs):
            x = self.share_net(obs)
            x = self.actor_net(x)
            mu = self.actor_mu_net(x)
            var = self.actor_var_net(x)
            return mu, var

    def __init__(
        self,
        observation_shape: Tuple[int, ...],
        n_out_actions: int,
        n_discrete_actions=None
    ):
        super(ContinuousBasicActorCriticSharedModel, self).__init__(
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
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(),
            nn.Linear(128, 1)
        )

        actor_model = ContinuousBasicActorCriticSharedModel.ActorModel(shared_net, actor_net, actor_mu_net, actor_var_net)
        critic_model = nn.Sequential(
            shared_net, critic_net
        )

        return actor_model, critic_model


@model_registry.add
class ContinuousBasicActorCriticEncoderSharedModel(DoubleModel):
    class ActorModel(nn.Module):
        def __init__(self, encoder_net, shared_net, actor_net, actor_mu_net, actor_var_net):
            super().__init__()
            self.encoder_net = encoder_net
            self.share_net = shared_net
            self.actor_net = actor_net
            self.actor_mu_net = actor_mu_net
            self.actor_var_net = actor_var_net

        def forward(self, obs):
            x = self.encoder_net(obs)
            x = self.share_net(x)
            x = self.actor_net(x)
            mu = self.actor_mu_net(x)
            var = self.actor_var_net(x)
            return mu, var

    def __init__(
        self,
        observation_shape: Tuple[int, ...],
        n_out_actions: int,
        n_discrete_actions=None,
        encoder_type=EncoderType.TWO_CONVOLUTION
    ):
        super(ContinuousBasicActorCriticEncoderSharedModel, self).__init__(
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
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1)),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(),
            )
        else:
            raise ValueError()

        encoder_out = self._get_conv_out(
            conv_layers=encoder_net,
            shape=self._observation_shape
        )

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
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(),
            nn.Linear(128, 1)
        )

        actor_model = ContinuousBasicActorCriticEncoderSharedModel.ActorModel(
            encoder_net, shared_net, actor_net, actor_mu_net, actor_var_net
        )
        critic_model = nn.Sequential(
            encoder_net,
            shared_net,
            critic_net
        )

        return actor_model, critic_model
