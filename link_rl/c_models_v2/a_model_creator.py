from abc import abstractmethod, ABC

from torch import nn
from typing import Tuple, final, Union

from link_rl.g_utils.registry import Registry


class ModelCreator(ABC):
    def __init__(
        self,
        n_input: int,
        n_out_actions: int,
        n_discrete_actions=None
    ):
        self._n_input = n_input
        self._n_out_actions = n_out_actions
        self._n_discrete_actions = n_discrete_actions

    @abstractmethod
    def _create_model(self) -> Union[nn.Module, Tuple[nn.Module, nn.Module]]:
        raise NotImplementedError

    @abstractmethod
    def create_model(self) -> Union[nn.Module, Tuple[nn.Module, nn.Module]]:
        raise NotImplementedError


model_creator_registry = Registry(ModelCreator)


class SingleModelCreator(ModelCreator, ABC):
    def __init__(
        self,
        n_input: int,
        n_out_actions: int,
        n_discrete_actions=None
    ):
        super(SingleModelCreator, self).__init__(
            n_input,
            n_out_actions,
            n_discrete_actions
        )

    @final
    def create_model(self) -> nn.Module:
        model = self._create_model()
        assert (isinstance(model, nn.Module),
                "self._create_model() has to return nn.Module")
        return model


class DoubleModelCreator(ModelCreator):
    def __init__(
        self,
        n_input: int,
        n_out_actions: int,
        n_discrete_actions=None
    ):
        super(DoubleModelCreator, self).__init__(
            n_input,
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

