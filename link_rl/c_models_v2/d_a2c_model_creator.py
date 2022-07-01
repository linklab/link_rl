from torch import nn
from typing import Tuple, final

from link_rl.c_models_v2.a_model_creator import DoubleModelCreator


class DiscreteActorCriticModelCreator(DoubleModelCreator):
    def __init__(
        self,
        n_input: int,
        n_out_actions: int,
        n_discrete_actions=None
    ):
        super(DiscreteActorCriticModelCreator, self).__init__(
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


class ContinuousActorCriticModelCreator(DoubleModelCreator):
    def __init__(
        self,
        n_input: int,
        n_out_actions: int,
        n_discrete_actions=None
    ):
        super(ContinuousActorCriticModelCreator, self).__init__(
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
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(),
            nn.Linear(128, 1)
        )

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

        actor_model = ActorModel()
        critic_model = nn.Sequential(
            shared_net, critic_net
        )

        return actor_model, critic_model
