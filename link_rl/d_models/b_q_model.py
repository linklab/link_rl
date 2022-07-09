import enum

import torch
from torch import nn
from typing import final, Tuple, List, Union, cast

from link_rl.d_models.a_model import SingleModel, model_registry


class Q_MODEL(enum.Enum):
    QModel = "QModel"
    DuelingQModel = "DuelingQModel"
    NatureAtariQModel = "NatureAtariQModel"


@model_registry.add
class QModel(SingleModel):
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


@model_registry.add
class DuelingQModel(SingleModel):
    class DuelingQModel(nn.Module):
        def __init__(self, shared_net, adv_net, val_net):
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

        dueling_q_model = DuelingQModel.DuelingQModel(shared_net, adv_net, val_net)
        return dueling_q_model


@model_registry.add
class NatureAtariQModel(SingleModel):
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
    def _create_model(self) -> nn.Module:
        linear_model = nn.Sequential(
            nn.Linear(self._n_input, 512),
            nn.LeakyReLU(),
            nn.Linear(512, self._n_discrete_actions)
        )

        model = nn.Sequential(
            linear_model
        )
        return model