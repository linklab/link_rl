import torch
from torch import nn
from typing import final, Tuple

from link_rl.c_models_v2.a_model_creator import SingleModelCreator, model_creator_registry


@model_creator_registry.add
class QModelCreator(SingleModelCreator):
    name = "QModelCreator"

    def __init__(
        self,
        observation_shape: Tuple[int, ...],
        n_out_actions: int,
        n_discrete_actions=None
    ):
        super(QModelCreator, self).__init__(
            observation_shape,
            n_out_actions,
            n_discrete_actions
        )

    @final
    def _create_model(self) -> nn.Module:
        model = nn.Sequential(
            nn.Linear(self._n_input, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(),
            nn.Linear(128, self._n_discrete_actions)
        )
        return model

@model_creator_registry.add
class DuelingQModelCreator(SingleModelCreator):
    name = "DuelingQModelCreator"
    def __init__(
        self,
        observation_shape: Tuple[int, ...],
        n_out_actions: int,
        n_discrete_actions=None
    ):
        super(DuelingQModelCreator, self).__init__(
            observation_shape,
            n_out_actions,
            n_discrete_actions
        )

    @final
    def _create_model(self) -> nn.Module:
        shared_net = nn.Sequential(
            nn.Linear(self._n_input, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU()
        )
        adv_net = nn.Linear(128, self._n_discrete_actions)
        val_net = nn.Linear(128, 1)

        class DuelingQModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.shared_net = shared_net
                self.adv_net = adv_net
                self.val_net = val_net

            def forward(self, obs):
                x = self.shared_net(obs)
                adv = self.adv_net(x)
                val = self.val_net(x)
                q_values = val + adv - torch.mean(adv, dim=-1, keepdim=True)

                return q_values

        dueling_q_model = DuelingQModel()
        return dueling_q_model
