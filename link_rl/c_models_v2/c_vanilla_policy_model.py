import enum

from torch import nn
from typing import final, Tuple
from link_rl.c_models_v2.a_model import SingleModel, model_registry


class BASIC_POLICY_MODEL(enum.Enum):
    DiscreteVanillaPolicyModel = "DiscreteVanillaPolicyModel"

@model_registry.add
class DiscreteVanillaPolicyModel(SingleModel):
    def __init__(
        self,
        observation_shape: Tuple[int, ...],
        n_out_actions: int,
        n_discrete_actions=None
    ):
        super(DiscreteVanillaPolicyModel, self).__init__(
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
            nn.Linear(128, self._n_discrete_actions),
            nn.Softmax(dim=-1)
        )
        return model

