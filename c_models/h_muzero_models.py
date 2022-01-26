from typing import Tuple
import torch
from torch import nn
import numpy as np
from a_configuration.b_base.c_models.convolutional_models import ParameterConvolutionalModel
from a_configuration.b_base.c_models.linear_models import ParameterLinearModel
from a_configuration.b_base.c_models.recurrent_convolutional_models import ParameterRecurrentConvolutionalModel
from a_configuration.b_base.c_models.recurrent_linear_models import ParameterRecurrentLinearModel
from c_models.a_models import Model
from c_models.c_policy_models import DiscreteActorModel, ContinuousActorModel
from torch.distributions import Normal, TanhTransform, TransformedDistribution
from e_main.parameter import parameter


class MuzeroModel:
    def __init__(
            self, observation_shape: Tuple[int], n_out_actions: int, n_discrete_actions=None, parameter=None,
            is_target_model=False
    ):
        pass

    def representation(self, observation):
        pass

    def dynamics(self):
        pass

    def prediction(self, encoded_state):
        pass

    def initial_inference(self, observation):
        encoded_state = self.representation(observation)
        policy_logits, value = self.prediction(encoded_state)
        # reward equal to 0 for consistency
        reward = torch.log(
            (
                torch.zeros(1, self.full_support_size)
                    .scatter(1, torch.tensor([[self.full_support_size // 2]]).long(), 1.0)
                    .repeat(len(observation), 1)
                    .to(observation.device)
            )
        )

        return value, reward, policy_logits, encoded_state

    def recurrent_inference(self, encoded_state, action):
        next_encoded_state, reward = self.dynamics(encoded_state, action)
        policy_logits, value = self.prediction(next_encoded_state)

        return value, reward, policy_logits, next_encoded_state
