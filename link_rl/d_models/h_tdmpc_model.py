import enum

import torch
from torch import nn
from typing import Tuple, final
from torch import distributions as pyd
from torch.distributions.utils import _standard_normal
from link_rl.d_models.a_model import SingleModel, model_registry
from collections import OrderedDict
import numpy as np


class TDMPC_MODEL(enum.Enum):
    TdmpcModel = "TdmpcModel"
    TdmpcModelParameterizedPlanAction = "TdmpcModelParameterizedPlanAction"


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
class TdmpcModel(SingleModel):
    class TOLDModel(nn.Module):
        def __init__(self, representation_net, dynamics_net, reward_net, pi_net, q1_net, q2_net):
            super().__init__()
            self.representation_net = representation_net
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
            z = self.representation(obs)
            a = self.pi_net(z)
            q1, q2 = self.Q(z, a)
            dynamics, reward = self.next(z, a)
            return q1, q2, dynamics, reward

        def track_q_grad(self, enable=True):
            """Utility function. Enables/disables gradient tracking of Q-networks."""
            for m in [self.q1_net, self.q2_net]:
                _set_requires_grad(m, enable)

        def representation(self, obs):
            """Encodes an observation into its latent representation (h)."""
            return self.representation_net(obs)

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
        representation_net = nn.Sequential(
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
            representation_net, dynamics_net, reward_net, pi_net, q1_net, q2_net
        )

        return told_model


@model_registry.add
class TdmpcModelParameterizedPlanAction(SingleModel):
    class TOLDModel(nn.Module):
        def __init__(
                self, representation_net, dynamics_net, reward_net, pi_net, q1_net, q2_net, policy_parameterized_net
                     ):
            super().__init__()
            self.representation_net = representation_net
            self.dynamics_net = dynamics_net
            self.reward_net = reward_net
            self.pi_net = pi_net
            self.q1_net = q1_net
            self.q2_net = q2_net
            self.policy_parameterized_net = policy_parameterized_net

            self.apply(_orthogonal_init)

            for m in [self.reward_net, self.q1_net, self.q2_net]:
                m[-1].weight.data.fill_(0)
                m[-1].bias.data.fill_(0)

        def policy_parameterized_pi(self, z):
            mu = torch.zeros(self._n_out_actions)
            z = self.pi_net(z)
            for i in range(len(self.policy_parameterized_net)):
                action_parameterized = self.policy_parameterized_net[i](z)
                mu[i] = action_parameterized
            return mu

        def forward(self, obs):
            z = self.representation(obs)
            a = self.policy_parameterized_pi(z)
            q1, q2 = self.Q(z, a)
            dynamics, reward = self.next(z, a)
            return q1, q2, dynamics, reward

        def track_q_grad(self, enable=True):
            """Utility function. Enables/disables gradient tracking of Q-networks."""
            for m in [self.q1_net, self.q2_net]:
                _set_requires_grad(m, enable)

        def representation(self, obs):
            """Encodes an observation into its latent representation (h)."""
            return self.representation_net(obs)

        def next(self, z, a):
            """Predicts next latent state (d) and single-step reward (R)."""
            x = torch.cat([z, a], dim=-1)
            return self.dynamics_net(x), self.reward_net(x)

        def pi(self, z, std=0):
            """Samples an action from the learned policy (pi)."""
            mu = torch.tanh(self.policy_parameterized_pi(z))
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
        representation_net = nn.Sequential(
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
            nn.ELU())

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

        policy_parameterized_net_dict = OrderedDict()
        for i in range(self._n_out_actions):
            policy_parameterized_net_dict["policy_parameterized_{0}".format(i)] = nn.Linear(512, 1)
        policy_parameterized_net = nn.Sequential(policy_parameterized_net_dict)

        told_model = TdmpcModelParameterizedPlanAction.TOLDModel(
            representation_net, dynamics_net, reward_net, pi_net, q1_net, q2_net, policy_parameterized_net
        )

        return told_model
