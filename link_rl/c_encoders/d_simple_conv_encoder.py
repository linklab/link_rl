from typing import Tuple

from torch import nn

from link_rl.c_encoders.a_encoder import BaseEncoder, encoder_registry


@encoder_registry.add
class SimpleConvEncoder(BaseEncoder):
    def __init__(self, observation_shape: Tuple[int, ...]):
        super().__init__(observation_shape)

    def _create_encoder(self) -> nn.Module:
        encoder = nn.Sequential(
            nn.Conv2d(in_channels=self._n_channels, out_channels=16, kernel_size=(4, 4), stride=(2, 2)),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(4, 4), stride=(2, 2)),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(4, 4), stride=(2, 2)),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1)),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Flatten(start_dim=1)
        )

        return encoder
