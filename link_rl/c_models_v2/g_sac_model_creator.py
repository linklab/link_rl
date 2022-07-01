import torch
from torch import nn
from typing import Tuple, final

from link_rl.c_models_v2.a_model_creator import DoubleModelCreator


class ContinuousSacModelCreator(DoubleModelCreator):
    def __init__(
        self,
        n_input: int,
        n_out_actions: int,
        n_discrete_actions=None
    ):
        super(ContinuousSacModelCreator, self).__init__(
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

        class ActorModel(nn.Module):
            def __init__(self):
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

        actor_model = ActorModel()
        critic_model = CriticModel()
        return actor_model, critic_model
