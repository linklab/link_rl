import enum

import torch
from torch import nn
from typing import Tuple, final
from torch import distributions as pyd
from torch.distributions.utils import _standard_normal
from link_rl.d_models.a_model import SingleModel, model_registry


class TDMPC_MODEL(enum.Enum):
    TdmpcEncoderModel = "TdmpcEncoderModel"
    TdmpcModel = "TdmpcModel"
    TdmpcRepresentationParameterizedEncoderModel = "TdmpcRepresentationParameterizedEncoderModel"
    TdmpcCNNParameterizedEncoderModel = "TdmpcCNNParameterizedEncoderModel"


class _TruncatedNormal(pyd.Normal):
    """Utility class implementing the truncated normal distribution."""

    def __init__(self, loc, scale, low=-1.0, high=1.0, eps=1e-6):
        super().__init__(loc, scale, validate_args=False)
        self.low = low
        self.high = high
        self.eps = eps

    def _clamp(self, x):
        clamped_x = torch.clamp(x, self.low + self.eps, self.high - self.eps)
        x = x - x.detach() + clamped_x.detach()
        return x

    def sample(self, clip=None, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        eps = _standard_normal(shape,
                               dtype=self.loc.dtype,
                               device=self.loc.device)
        eps *= self.scale
        if clip is not None:
            eps = torch.clamp(eps, -clip, clip)
        x = self.loc + eps
        return self._clamp(x)


def _orthogonal_init(m):
    """Orthogonal layer initialization."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data, gain)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def _set_requires_grad(net, value):
    """Enable/disable gradients for a given (sub)network."""
    for param in net.parameters():
        param.requires_grad_(value)


@model_registry.add
class TdmpcEncoderModel(SingleModel):
    class TOLDModel(nn.Module):
        def __init__(self, encoder_net, dynamics_net, reward_net, pi_net, q1_net, q2_net):
            super().__init__()
            self.encoder_net = encoder_net
            self.dynamics_net = dynamics_net
            self.reward_net = reward_net
            self.pi_net = pi_net
            self.q1_net = q1_net
            self.q2_net = q2_net

            self.apply(_orthogonal_init)

            for m in [self.reward_net, self.q1_net, self.q2_net]:
                m[-1].weight.data.fill_(0)
                m[-1].bias.data.fill_(0)

        def track_q_grad(self, enable=True):
            """Utility function. Enables/disables gradient tracking of Q-networks."""
            for m in [self.q1_net, self.q2_net]:
                _set_requires_grad(m, enable)

        def encode(self, obs):
            """Encodes an observation into its latent representation (h)."""
            return self.encoder_net(obs)

        def next(self, z, a):
            """Predicts next latent state (d) and single-step reward (R)."""
            x = torch.cat([z, a], dim=-1)
            return self.dynamics_net(x), self.reward_net(x)

        def pi(self, z, std=0):
            """Samples an action from the learned policy (pi)."""
            mu = torch.tanh(self.pi_net(z))
            if std > 0:
                std = torch.ones_like(mu) * std
                return _TruncatedNormal(mu, std).sample(clip=0.3)
            return mu

        def Q(self, z, a):
            """Predict state-action value (Q)."""
            x = torch.cat([z, a], dim=-1)
            return self.q1_net(x), self.q2_net(x)

    def __init__(
        self,
        n_input: int,
        n_out_actions: int,
        n_discrete_actions=None
    ):
        super().__init__(
            n_input,
            n_out_actions,
            n_discrete_actions
        )

    @final
    def _create_model(self) -> Tuple[nn.Module, nn.Module]:
        cnn_net = nn.Sequential(
            nn.Conv2d(self._n_input, 32, 7, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, 5, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=2),
            nn.ReLU()
        )
        cnn_out = self._get_encoder_out(cnn_net, self._observation_shape)
        representation_net = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(cnn_out, 50)
        )
        encoder_net = nn.Sequential(
            cnn_net,
            representation_net
        )

        dynamics_net = nn.Sequential(
            nn.Linear(50 + self._n_out_actions, 512),
            nn.ELU(),
            nn.Linear(512, 512),
            nn.ELU(),
            nn.Linear(512, 50))

        reward_net = nn.Sequential(
            nn.Linear(50 + self._n_out_actions, 512),
            nn.ELU(),
            nn.Linear(512, 512),
            nn.ELU(),
            nn.Linear(512, 1))

        pi_net = nn.Sequential(
            nn.Linear(50, 512),
            nn.ELU(),
            nn.Linear(512, 512),
            nn.ELU(),
            nn.Linear(512, self._n_out_actions))

        q1_net = nn.Sequential(nn.Linear(50 + self._n_out_actions, 512),
                               nn.LayerNorm(512),
                               nn.Tanh(),
                               nn.Linear(512, 512), nn.ELU(),
                               nn.Linear(512, 1))
        q2_net = nn.Sequential(nn.Linear(50 + self._n_out_actions, 512),
                               nn.LayerNorm(512),
                               nn.Tanh(),
                               nn.Linear(512, 512), nn.ELU(),
                               nn.Linear(512, 1))

        told_model = TdmpcEncoderModel.TOLDModel(
            encoder_net, dynamics_net, reward_net, pi_net, q1_net, q2_net
        )

        return told_model


@model_registry.add
class TdmpcModel(SingleModel):
    class TOLDModel(nn.Module):
        def __init__(self, encoder_net, dynamics_net, reward_net, pi_net, q1_net, q2_net):
            super().__init__()
            self.encoder_net = encoder_net
            self.dynamics_net = dynamics_net
            self.reward_net = reward_net
            self.pi_net = pi_net
            self.q1_net = q1_net
            self.q2_net = q2_net

            self.apply(_orthogonal_init)

            for m in [self.reward_net, self.q1_net, self.q2_net]:
                m[-1].weight.data.fill_(0)
                m[-1].bias.data.fill_(0)

        def forward(self, obs):
            z = self.encoder_net(obs)
            a = self.pi_net(z)
            q1 = self.q1_net(z, a)
            q2 = self.q2_net(z, a)
            dynamics = self.dynamics_net(z, a)
            reward = self.reward_net(z, a)
            return q1, q2, dynamics, reward

        def track_q_grad(self, enable=True):
            """Utility function. Enables/disables gradient tracking of Q-networks."""
            for m in [self.q1_net, self.q2_net]:
                _set_requires_grad(m, enable)

        def encode(self, obs):
            """Encodes an observation into its latent representation (h)."""
            return self.encoder_net(obs)

        def next(self, z, a):
            """Predicts next latent state (d) and single-step reward (R)."""
            x = torch.cat([z, a], dim=-1)
            return self.dynamics_net(x), self.reward_net(x)

        def pi(self, z, std=0):
            """Samples an action from the learned policy (pi)."""
            mu = torch.tanh(self.pi_net(z))
            if std > 0:
                std = torch.ones_like(mu) * std
                return _TruncatedNormal(mu, std).sample(clip=0.3)
            return mu

        def Q(self, z, a):
            """Predict state-action value (Q)."""
            x = torch.cat([z, a], dim=-1)
            return self.q1_net(x), self.q2_net(x)

    def __init__(
        self,
        n_input: int,
        n_out_actions: int,
        n_discrete_actions=None
    ):
        super().__init__(
            n_input,
            n_out_actions,
            n_discrete_actions
        )

    @final
    def _create_model(self) -> Tuple[nn.Module, nn.Module]:
        encoder_net = nn.Sequential(
            nn.Linear(self._n_input, 256),
            nn.ELU(),
            nn.Linear(256, 50)
        )

        dynamics_net = nn.Sequential(
            nn.Linear(50 + self._n_out_actions, 512),
            nn.ELU(),
            nn.Linear(512, 512),
            nn.ELU(),
            nn.Linear(512, 50))

        reward_net = nn.Sequential(
            nn.Linear(50 + self._n_out_actions, 512),
            nn.ELU(),
            nn.Linear(512, 512),
            nn.ELU(),
            nn.Linear(512, 1))

        pi_net = nn.Sequential(
            nn.Linear(50, 512),
            nn.ELU(),
            nn.Linear(512, 512),
            nn.ELU(),
            nn.Linear(512, self._n_out_actions))

        q1_net = nn.Sequential(nn.Linear(50 + self._n_out_actions, 512),
                               nn.LayerNorm(512),
                               nn.Tanh(),
                               nn.Linear(512, 512), nn.ELU(),
                               nn.Linear(512, 1))
        q2_net = nn.Sequential(nn.Linear(50 + self._n_out_actions, 512),
                               nn.LayerNorm(512),
                               nn.Tanh(),
                               nn.Linear(512, 512), nn.ELU(),
                               nn.Linear(512, 1))

        told_model = TdmpcModel.TOLDModel(
            encoder_net, dynamics_net, reward_net, pi_net, q1_net, q2_net
        )

        return told_model


@model_registry.add
class TdmpcRepresentationParameterizedEncoderModel(SingleModel):
    class TOLDModel(nn.Module):
        def __init__(
                self, cnn_net, representation_net, coef_parameterized_net, freedom_parameterized_net,
                dynamics_net, reward_net, pi_net, q1_net, q2_net
        ):
            super().__init__()
            #######################ENCODER############################
            self.cnn_net = cnn_net
            self.representation_net = representation_net
            self.coef_parameterized_net = coef_parameterized_net
            self.freedom_parameterized_net = freedom_parameterized_net
            ##########################################################
            self.dynamics_net = dynamics_net
            self.reward_net = reward_net
            self.pi_net = pi_net
            self.q1_net = q1_net
            self.q2_net = q2_net

            self.apply(_orthogonal_init)

            for m in [self.reward_net, self.q1_net, self.q2_net]:
                m[-1].weight.data.fill_(0)
                m[-1].bias.data.fill_(0)

        def track_q_grad(self, enable=True):
            """Utility function. Enables/disables gradient tracking of Q-networks."""
            for m in [self.q1_net, self.q2_net]:
                _set_requires_grad(m, enable)

        def encode(self, obs):
            """Encodes an observation into its latent representation (h)."""
            x = self.cnn_net(obs)
            z = self.representation_net(x)
            k = self.coef_parameterized_net(z)
            z0 = self.freedom_parameterized_net(z)
            return (k*z) - (k*z0)

        def next(self, z, a):
            """Predicts next latent state (d) and single-step reward (R)."""
            x = torch.cat([z, a], dim=-1)
            return self.dynamics_net(x), self.reward_net(x)

        def pi(self, z, std=0):
            """Samples an action from the learned policy (pi)."""
            mu = torch.tanh(self.pi_net(z))
            if std > 0:
                std = torch.ones_like(mu) * std
                return _TruncatedNormal(mu, std).sample(clip=0.3)
            return mu

        def Q(self, z, a):
            """Predict state-action value (Q)."""
            x = torch.cat([z, a], dim=-1)
            return self.q1_net(x), self.q2_net(x)

    def __init__(
        self,
        n_input: int,
        n_out_actions: int,
        n_discrete_actions=None
    ):
        super().__init__(
            n_input,
            n_out_actions,
            n_discrete_actions
        )

    @final
    def _create_model(self) -> Tuple[nn.Module, nn.Module]:
        cnn_net = nn.Sequential(
            nn.Conv2d(self._n_input, 32, 7, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, 5, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=2),
            nn.ReLU()
        )
        cnn_out = self._get_encoder_out(cnn_net, self._observation_shape)
        representation_net = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(cnn_out, 50)
        )

        coef_parameterized_net = nn.Sequential(
            nn.Linear(50, 50),
            nn.Sigmoid()
        )

        freedom_parameterized_net = nn.Sequential(
            nn.Linear(50, 50)
        )

        dynamics_net = nn.Sequential(
            nn.Linear(50 + self._n_out_actions, 512),
            nn.ELU(),
            nn.Linear(512, 512),
            nn.ELU(),
            nn.Linear(512, 50))

        reward_net = nn.Sequential(
            nn.Linear(50 + self._n_out_actions, 512),
            nn.ELU(),
            nn.Linear(512, 512),
            nn.ELU(),
            nn.Linear(512, 1))

        pi_net = nn.Sequential(
            nn.Linear(50, 512),
            nn.ELU(),
            nn.Linear(512, 512),
            nn.ELU(),
            nn.Linear(512, self._n_out_actions))

        q1_net = nn.Sequential(nn.Linear(50 + self._n_out_actions, 512),
                               nn.LayerNorm(512),
                               nn.Tanh(),
                               nn.Linear(512, 512), nn.ELU(),
                               nn.Linear(512, 1))
        q2_net = nn.Sequential(nn.Linear(50 + self._n_out_actions, 512),
                               nn.LayerNorm(512),
                               nn.Tanh(),
                               nn.Linear(512, 512), nn.ELU(),
                               nn.Linear(512, 1))

        told_model = TdmpcRepresentationParameterizedEncoderModel.TOLDModel(
            cnn_net, representation_net, coef_parameterized_net, freedom_parameterized_net, dynamics_net,
            reward_net, pi_net, q1_net, q2_net
        )

        return told_model


@model_registry.add
class TdmpcCNNParameterizedEncoderModel(SingleModel):
    class TOLDModel(nn.Module):
        def __init__(
                self, cnn_net, representation_net, coef_parameterized_net, freedom_parameterized_net,
                dynamics_net, reward_net, pi_net, q1_net, q2_net
        ):
            super().__init__()
            #######################ENCODER############################
            self.cnn_net = cnn_net
            self.representation_net = representation_net
            self.coef_parameterized_net = coef_parameterized_net
            self.freedom_parameterized_net = freedom_parameterized_net
            ##########################################################
            self.dynamics_net = dynamics_net
            self.reward_net = reward_net
            self.pi_net = pi_net
            self.q1_net = q1_net
            self.q2_net = q2_net

            self.apply(_orthogonal_init)

            for m in [self.reward_net, self.q1_net, self.q2_net]:
                m[-1].weight.data.fill_(0)
                m[-1].bias.data.fill_(0)

        def track_q_grad(self, enable=True):
            """Utility function. Enables/disables gradient tracking of Q-networks."""
            for m in [self.q1_net, self.q2_net]:
                _set_requires_grad(m, enable)

        def encode(self, obs):
            """Encodes an observation into its latent representation (h)."""
            x = self.cnn_net(obs)
            z = self.representation_net(x)
            k = self.coef_parameterized_net(x)
            z0 = self.freedom_parameterized_net(x)
            return (k*z) - (k*z0)

        def next(self, z, a):
            """Predicts next latent state (d) and single-step reward (R)."""
            x = torch.cat([z, a], dim=-1)
            return self.dynamics_net(x), self.reward_net(x)

        def pi(self, z, std=0):
            """Samples an action from the learned policy (pi)."""
            mu = torch.tanh(self.pi_net(z))
            if std > 0:
                std = torch.ones_like(mu) * std
                return _TruncatedNormal(mu, std).sample(clip=0.3)
            return mu

        def Q(self, z, a):
            """Predict state-action value (Q)."""
            x = torch.cat([z, a], dim=-1)
            return self.q1_net(x), self.q2_net(x)

    def __init__(
        self,
        n_input: int,
        n_out_actions: int,
        n_discrete_actions=None
    ):
        super().__init__(
            n_input,
            n_out_actions,
            n_discrete_actions
        )

    @final
    def _create_model(self) -> Tuple[nn.Module, nn.Module]:
        cnn_net = nn.Sequential(
            nn.Conv2d(self._n_input, 32, 7, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, 5, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=2),
            nn.ReLU()
        )
        cnn_out = self._get_encoder_out(cnn_net, self._observation_shape)
        representation_net = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(cnn_out, 50)
        )

        coef_parameterized_net = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(cnn_out, 50),
            nn.Sigmoid()
        )

        freedom_parameterized_net = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(cnn_out, 50)
        )

        dynamics_net = nn.Sequential(
            nn.Linear(50 + self._n_out_actions, 512),
            nn.ELU(),
            nn.Linear(512, 512),
            nn.ELU(),
            nn.Linear(512, 50))

        reward_net = nn.Sequential(
            nn.Linear(50 + self._n_out_actions, 512),
            nn.ELU(),
            nn.Linear(512, 512),
            nn.ELU(),
            nn.Linear(512, 1))

        pi_net = nn.Sequential(
            nn.Linear(50, 512),
            nn.ELU(),
            nn.Linear(512, 512),
            nn.ELU(),
            nn.Linear(512, self._n_out_actions))

        q1_net = nn.Sequential(nn.Linear(50 + self._n_out_actions, 512),
                               nn.LayerNorm(512),
                               nn.Tanh(),
                               nn.Linear(512, 512), nn.ELU(),
                               nn.Linear(512, 1))
        q2_net = nn.Sequential(nn.Linear(50 + self._n_out_actions, 512),
                               nn.LayerNorm(512),
                               nn.Tanh(),
                               nn.Linear(512, 512), nn.ELU(),
                               nn.Linear(512, 1))

        told_model = TdmpcCNNParameterizedEncoderModel.TOLDModel(
            cnn_net, representation_net, coef_parameterized_net, freedom_parameterized_net, dynamics_net,
            reward_net, pi_net, q1_net, q2_net
        )

        return told_model
