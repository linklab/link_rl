import enum

import torch
from torch import nn
from typing import Tuple, final

from link_rl.d_models.a_model import DoubleModel, model_registry


class TD3_MODEL(enum.Enum):
    ContinuousTd3Model = "ContinuousTd3Model"


@model_registry.add
class ContinuousTd3Model(DoubleModel):
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
        shared_net = nn.Sequential(
            nn.Linear(self._n_input, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(),
        )

        actor_net = nn.Sequential(
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(),
            nn.Linear(128, self._n_out_actions),
            nn.Tanh()
        )
        critic_net = nn.Sequential(
            nn.Linear(128 + self._n_out_actions, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(),
        )
        q1_critic_net = nn.Linear(128, 1)
        q2_critic_net = nn.Linear(128, 1)

        actor_model = nn.Sequential(
            shared_net, actor_net
        )
        critic_model = ContinuousTd3Model.CriticModel(shared_net, critic_net, q1_critic_net, q2_critic_net)

        return actor_model, critic_model




