from torch import nn
from typing import final
from link_rl.c_models_v2.a_model_creator import SingleModelCreator, model_creator_registry


@model_creator_registry.add
class DiscretePolicyModelCreator(SingleModelCreator):
    name = "DiscretePolicyModelCreator"

    def __init__(
        self,
        n_input: int,
        n_out_actions: int,
        n_discrete_actions=None
    ):
        super(DiscretePolicyModelCreator, self).__init__(
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

