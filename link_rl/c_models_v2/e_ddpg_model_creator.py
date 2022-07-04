import torch
from torch import nn
from typing import Tuple, final

from link_rl.c_models_v2.a_model_creator import DoubleModelCreator, model_creator_registry


@model_creator_registry.add
class ContinuousDdpgModelCreator(DoubleModelCreator):
    name = "ContinuousDdpgModelCreator"

    class CriticModel(nn.Module):
        def __init__(self, shared_net, critic_net):
            super().__init__()
            self.shared_net = shared_net
            self.critic_net = critic_net

        def forward(self, obs, action):
            x = self.shared_net(obs)
            x = torch.cat([x, action], dim=-1).float()
            q = self.critic_net(x)
            return q

    def __init__(
        self,
        observation_shape: Tuple[int, ...],
        n_out_actions: int,
        n_discrete_actions=None
    ):
        super(ContinuousDdpgModelCreator, self).__init__(
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
            nn.Linear(128, self._n_out_actions),
            nn.Tanh()
        )
        critic_net = nn.Sequential(
            nn.Linear(128 + self._n_out_actions, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(),
            nn.Linear(128, 1)
        )

        actor_model = nn.Sequential(
            shared_net, actor_net
        )
        critic_model = ContinuousDdpgModelCreator.CriticModel(shared_net, critic_net)

        return actor_model, critic_model
