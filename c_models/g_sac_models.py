from typing import Tuple
import torch

from a_configuration.b_base.c_models.convolutional_models import ParameterConvolutionalModel
from a_configuration.b_base.c_models.linear_models import ParameterLinearModel
from a_configuration.b_base.c_models.recurrent_models import ParameterRecurrentModel
from c_models.a_models import Model


class SacModel(Model):
    def __init__(
            self, observation_shape: Tuple[int], n_out_actions: int, device=torch.device("cpu"), parameter=None
    ):
        super(SacModel, self).__init__(observation_shape, n_out_actions, device, parameter)
        if isinstance(self.parameter.MODEL, ParameterLinearModel):
            pass
        elif isinstance(self.parameter.MODEL, ParameterConvolutionalModel):
            pass
        elif isinstance(self.parameter.MODEL, ParameterRecurrentModel):
            pass
        else:
            raise ValueError()