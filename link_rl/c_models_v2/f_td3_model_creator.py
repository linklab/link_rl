import torch
from torch import nn
from typing import Tuple, final

from link_rl.c_models_v2.a_model_creator import DoubleModelCreator, model_creator_registry


@model_creator_registry.add
class ContinuousTd3ModelCreator(DoubleModelCreator):
    name = "ContinuousTd3ModelCreator"

    def __init__(
        self,
        observation_shape: Tuple[int, ...],
        n_out_actions: int,
        n_discrete_actions=None
    ):
        super(ContinuousTd3ModelCreator, self).__init__(
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
            nn.Softmax(dim=-1)
        )
        critic_net = nn.Sequential(
            nn.Linear(128 + self._n_out_actions, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(),
        )
        q1_critic_net = nn.Linear(128, 1)
        q2_critic_net = nn.Linear(128, 1)

        class CriticModel(nn.Module):
            def __init__(self):
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

        actor_model = nn.Sequential(
            shared_net, actor_net
        )
        critic_model = CriticModel()

        return actor_model, critic_model
