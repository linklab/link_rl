from typing import Tuple
import torch
from torch import nn
import numpy as np

from link_rl.c_encoders.a_encoder import BaseEncoder, encoder_registry


@encoder_registry.add
class SimpleTdmpcEncoder(BaseEncoder):
    def __init__(self, observation_shape: Tuple[int, ...]):
        super().__init__(observation_shape)

    def _create_encoder(self) -> nn.Module:
        conv_layers = self._create_conv()
        conv_out = self.get_encoder_out(conv_layers, self._observation_shape)

        representation_layers = nn.Linear(conv_out, 50)

        encoder = nn.Sequential(conv_layers, representation_layers)

        return encoder

    def _create_conv(self) -> nn.Module:
        conv_layers = nn.Sequential(
            nn.Conv2d(self._n_channels, 32, 7, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, 5, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=2),
            nn.ReLU(),
            nn.Flatten(start_dim=1)
        )

        return conv_layers
