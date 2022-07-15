from abc import abstractmethod, ABC

import enum
import numpy as np
import torch
from torch import nn
from typing import Tuple, final, Optional

from link_rl.h_utils.registry import Registry


class ENCODER(enum.Enum):
    IdentityEncoder = "IdentityEncoder"
    SimpleConvEncoder = "SimpleConvEncoder"
    NatureAtariEncoder = "NatureAtariEncoder"
    SimpleTdmpcEncoder = "SimpleTdmpcEncoder"


class BaseEncoder(ABC):
    def __init__(self, observation_shape: Tuple[int, ...]):
        self._observation_shape = observation_shape
        self._n_channels = observation_shape[0]

        encoder = self._create_encoder()
        self._encoder_out = self.get_encoder_out(encoder, self._observation_shape)
        del encoder

    def get_encoder_out(self, conv_layers, shape) -> int:
        conv_layers.eval()
        encoder_out = conv_layers(torch.zeros(1, *shape))
        return encoder_out

    @abstractmethod
    def _create_encoder(self) -> nn.Module:
        """
        The output of encoder must be flattened
        """
        raise NotImplementedError

    @final
    def create_encoder(self) -> nn.Module:
        encoder = self._create_encoder()
        return encoder

    @property
    def encoder_out(self):
        return np.asarray((3,40,40))


encoder_registry = Registry(BaseEncoder)
