from abc import abstractmethod, ABC

import numpy as np
import torch
from torch import nn
from typing import Tuple, final, Optional

from link_rl.g_utils.registry import Registry


class BaseEncoder(ABC):
    def __init__(self, observation_shape: Tuple[int, ...]):
        self._observation_shape = observation_shape
        self._n_channels = observation_shape[0]

        def get_conv_out(conv_layers, shape) -> int:
            conv_layers.eval()
            conv_out = conv_layers(torch.zeros(1, *shape))
            return int(np.prod(conv_out.size()))
        encoder = self._create_encoder()
        self._conv_out = get_conv_out(encoder, self._observation_shape)
        del encoder

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
    def conv_out(self):
        return self._conv_out


encoder_registry = Registry(BaseEncoder)
