from abc import abstractmethod, ABC

import numpy as np
import torch
from torch import nn
from typing import Tuple, final, Union

from link_rl.g_utils.registry import Registry


class BaseModel(ABC):
    def __init__(
        self,
        observation_shape: Tuple[int, ...],
        n_out_actions: int,
        n_discrete_actions=None
    ):
        self._observation_shape = observation_shape
        self._n_input = observation_shape[0]
        self._n_out_actions = n_out_actions
        self._n_discrete_actions = n_discrete_actions

    @abstractmethod
    def _create_model(self) -> Union[nn.Module, Tuple[nn.Module, nn.Module]]:
        raise NotImplementedError

    @abstractmethod
    def create_model(self) -> Union[nn.Module, Tuple[nn.Module, nn.Module]]:
        raise NotImplementedError

    @staticmethod
    def _get_conv_out(conv_layers, shape):
        conv_out = conv_layers(torch.zeros(1, *shape))
        return int(np.prod(conv_out.size()))


model_registry = Registry(BaseModel)


class SingleModel(BaseModel, ABC):
    def __init__(
        self,
        observation_shape: Tuple[int, ...],
        n_out_actions: int,
        n_discrete_actions=None
    ):
        super(SingleModel, self).__init__(
            observation_shape,
            n_out_actions,
            n_discrete_actions
        )

    @final
    def create_model(self) -> nn.Module:
        model = self._create_model()
        assert (isinstance(model, nn.Module),
                "self._create_model() has to return nn.Module")
        return model


class DoubleModel(BaseModel):
    def __init__(
        self,
        observation_shape: Tuple[int, ...],
        n_out_actions: int,
        n_discrete_actions=None
    ):
        super(DoubleModel, self).__init__(
            observation_shape,
            n_out_actions,
            n_discrete_actions
        )

    @final
    def create_model(self) -> Tuple[nn.Module, nn.Module]:
        actor_critic_model = self._create_model()
        assert (isinstance(actor_critic_model, Tuple),
                "self._create_actor_critic_model() has to return Tuple[nn.Module, nn.Module]")
        actor_model, critic_model = actor_critic_model
        assert (isinstance(actor_model, nn.Module),
                "self._create_actor_critic_model() has to return Tuple[nn.Module, nn.Module]")
        assert (isinstance(critic_model, nn.Module),
                "self._create_actor_critic_model() has to return Tuple[nn.Module, nn.Module]")
        return actor_model, critic_model

