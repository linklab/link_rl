from typing import Tuple
import torch

from a_configuration.b_base.c_models.convolutional_models import ParameterConvolutionalModel
from a_configuration.b_base.c_models.linear_models import ParameterLinearModel
from a_configuration.b_base.c_models.recurrent_models import ParameterRecurrentModel
from c_models.a_models import Model
from c_models.c_policy_models import DiscreteActorModel, ContinuousActorModel


class SacCriticModel(Model):
    def __init__(
            self, observation_shape: Tuple[int], n_out_actions: int, n_discrete_actions=None,
            device=torch.device("cpu"), parameter=None
    ):
        super(SacCriticModel, self).__init__(observation_shape, n_out_actions, n_discrete_actions, device, parameter)

        # TwinQ Models
        if isinstance(self.parameter.MODEL, ParameterLinearModel):
            pass
        elif isinstance(self.parameter.MODEL, ParameterConvolutionalModel):
            pass
        elif isinstance(self.parameter.MODEL, ParameterRecurrentModel):
            pass
        else:
            raise ValueError()


class DiscreteSacModel(DiscreteActorModel, SacCriticModel):
    def __init__(
            self, observation_shape: Tuple[int], n_out_actions: int, n_discrete_actions=None,
            device=torch.device("cpu"), parameter=None
    ):
        super(DiscreteSacModel, self).__init__(observation_shape, n_out_actions, n_discrete_actions, device, parameter)


class ContinuousSacModel(ContinuousActorModel, SacCriticModel):
    def __init__(
            self, observation_shape: Tuple[int], n_out_actions: int, device=torch.device("cpu"), parameter=None
    ):
        super(ContinuousSacModel, self).__init__(observation_shape, n_out_actions, device, parameter)