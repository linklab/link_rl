import enum

from torch import nn
from typing import final
from link_rl.d_models.a_model import SingleModel, model_registry


class VANILLA_POLICY_MODEL(enum.Enum):
    DiscreteVanillaPolicyModel = "DiscreteVanillaPolicyModel"


@model_registry.add
class DiscreteVanillaPolicyModel(SingleModel):
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
            nn.Linear(128, self._n_discrete_actions),
            nn.Softmax(dim=-1)
        )
        return model
