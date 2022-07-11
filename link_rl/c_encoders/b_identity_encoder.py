from typing import Tuple

from torch import nn

from link_rl.c_encoders.a_encoder import BaseEncoder, encoder_registry


@encoder_registry.add
class IdentityEncoder(BaseEncoder):
    def __init__(self, observation_shape: Tuple[int, ...]):
        super().__init__(observation_shape)

    def _create_encoder(self) -> nn.Module:
        return nn.Identity()