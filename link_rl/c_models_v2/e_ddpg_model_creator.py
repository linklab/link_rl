import torch
from torch import nn
from typing import Tuple, final

from link_rl.c_models_v2.a_model_creator import DoubleModelCreator


class ContinuousDdpgModelCreator(DoubleModelCreator):
    def __init__(
        self,
        n_input: int,
        n_out_actions: int,
        n_discrete_actions=None
    ):
        super(ContinuousDdpgModelCreator, self).__init__(
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
            nn.Softmax(dim=-1)
        )
        critic_net = nn.Sequential(
            nn.Linear(128 + self._n_out_actions, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(),
            nn.Linear(128, 1)
        )

        class CriticModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.shared_net = shared_net
                self.critic_net = critic_net

            def forward(self, obs, action):
                x = self.shared_net(obs)
                x = torch.cat([x, action], dim=-1).float()
                q = self.critic_net(x)
                return q

        actor_model = nn.Sequential(
            shared_net, actor_net
        )
        critic_model = CriticModel()

        return actor_model, critic_model
