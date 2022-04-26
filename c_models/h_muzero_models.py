from typing import Tuple
import torch
from torch import nn

from a_configuration.a_base_config.c_models.config_convolutional_models import Config2DConvolutionalModel
from a_configuration.a_base_config.c_models.config_linear_models import ConfigLinearModel
from c_models.a_models import Model


class DiscreteMuzeroModel(Model):
    def __init__(
            self,
            observation_shape: Tuple[int],
            n_out_actions: int,
            n_discrete_actions=None,
            config=None
    ):
        super(DiscreteMuzeroModel, self).__init__(
            observation_shape=observation_shape,
            n_out_actions=n_out_actions,
            n_discrete_actions=n_discrete_actions,
            config=config
        )
        if isinstance(self.config.MODEL_PARAMETER, ConfigLinearModel):
            # representation_layers
            input_n_features = observation_shape[0]
            self.representation_network = self.get_representation_layers(input_n_features=input_n_features)

            assert self.config.MODEL_PARAMETER.NEURONS_PER_REPRESENTATION_LAYER is not None
            input_n_features = self.config.MODEL_PARAMETER.NEURONS_PER_REPRESENTATION_LAYER[-1]

            assert n_out_actions is not None
            input_n_features_with_n_out_actions = input_n_features + self.n_out_actions  # self.n_out_actions = 1

            self.dynamics_encoded_state_network = self.get_linear_layers(
                input_n_features=input_n_features_with_n_out_actions
            )

            self.dynamics_reward_layers = self.get_linear_layers(input_n_features=input_n_features)
            self.prediction_policy_layers = self.get_linear_layers(input_n_features=input_n_features)
            self.prediction_value_layers = self.get_linear_layers(input_n_features=input_n_features)

        elif isinstance(self.config.MODEL_PARAMETER, Config2DConvolutionalModel):
            # convolutional_layers
            input_n_channels = observation_shape[0]
            self.convolutional_layers = self.get_convolutional_layers(input_n_channels)

            # representation_layers
            conv_out_flat_size = self._get_conv_out(self.convolutional_layers, observation_shape)
            self.representation_network = self.get_representation_layers(input_n_features=conv_out_flat_size)

            assert self.config.MODEL_PARAMETER.NEURONS_PER_REPRESENTATION_LAYER is not None
            input_n_features = self.config.MODEL_PARAMETER.NEURONS_PER_REPRESENTATION_LAYER[-1]

            assert n_out_actions is not None
            input_n_features_with_n_out_actions = input_n_features + self.n_out_actions  # self.n_out_actions = 1

            self.dynamics_encoded_state_network = self.get_linear_layers(
                input_n_features=input_n_features_with_n_out_actions
            )

            self.dynamics_reward_layers = self.get_linear_layers(input_n_features=input_n_features)
            self.prediction_policy_layers = self.get_linear_layers(input_n_features=input_n_features)
            self.prediction_value_layers = self.get_linear_layers(input_n_features=input_n_features)
        else:
            raise ValueError()

        self.dynamics_reward_last_layer = nn.Linear(
            self.config.MODEL_PARAMETER.NEURONS_PER_FULLY_CONNECTED_LAYER[-1], self.config.SUPPORT_SIZE * 2 + 1
        )

        self.prediction_last_policy_layer = nn.Linear(
            self.config.MODEL_PARAMETER.NEURONS_PER_FULLY_CONNECTED_LAYER[-1], self.n_discrete_actions
        )

        self.prediction_value_last_layer = nn.Linear(
            self.config.MODEL_PARAMETER.NEURONS_PER_FULLY_CONNECTED_LAYER[-1], self.config.SUPPORT_SIZE * 2 + 1
        )

        self.critic_params_list = list(self.parameters())

    def representation(self, observation):
        encoded_state = self.representation_network(observation)

        return encoded_state


    def dynamics(self, encoded_state, action):
        # action_one_hot = (
        #     torch.zeros((action.shape[0], self.n_discrete_actions))
        #         .to(self.config.DEVICE)
        #         .float()
        # )
        # action_one_hot.scatter_(1, action.long(), 1.0)

        x = torch.cat((encoded_state, action), dim=1)
        next_encoded_state = self.dynamics_encoded_state_network(x)

        x = self.dynamics_reward_layers(next_encoded_state)
        reward = self.dynamics_reward_last_layer(x)

        return next_encoded_state, reward

    def prediction(self, encoded_state):
        x = self.prediction_policy_layers(encoded_state)
        policy_logits = self.prediction_last_policy_layer(x)

        x = self.prediction_value_layers(encoded_state)
        value = self.prediction_value_last_layer(x)

        return policy_logits, value

    def initial_inference(self, observation):
        encoded_state = self.representation(observation)
        policy_logits, value = self.prediction(encoded_state)
        # reward equal to 0 for consistency
        reward = torch.log(
            (
                torch.zeros(1, self.config.SUPPORT_SIZE * 2 + 1)
                    .scatter(1, torch.tensor([[self.config.SUPPORT_SIZE]]).long(), 1.0)
                    .repeat(len(observation), 1)
                    .to(observation.device)
            )
        )

        return value, reward, policy_logits, encoded_state

    def recurrent_inference(self, encoded_state, action):
        next_encoded_state, reward = self.dynamics(encoded_state, action)
        policy_logits, value = self.prediction(next_encoded_state)

        return value, reward, policy_logits, next_encoded_state


def scalar_to_support(x, support_size):
    """
    Transform a scalar to a categorical representation with (2 * support_size + 1) categories
    See paper appendix Network Architecture
    """
    # Reduce the scale (defined in https://arxiv.org/abs/1805.11593)
    x = torch.sign(x) * (torch.sqrt(torch.abs(x) + 1) - 1) + 0.001 * x

    # Encode on a vector
    x = torch.clamp(x, -support_size, support_size)
    floor = x.floor()
    prob = x - floor
    logits = torch.zeros(x.shape[0], x.shape[1], 2 * support_size + 1).to(x.device)
    logits.scatter_(
        2, (floor + support_size).long().unsqueeze(-1), (1 - prob).unsqueeze(-1)
    )
    indexes = floor + support_size + 1
    prob = prob.masked_fill_(2 * support_size < indexes, 0.0)
    indexes = indexes.masked_fill_(2 * support_size < indexes, 0.0)
    logits.scatter_(2, indexes.long().unsqueeze(-1), prob.unsqueeze(-1))
    return logits

def support_to_scalar(logits, support_size):
    """
    Transform a categorical representation to a scalar
    See paper appendix Network Architecture
    """
    # Decode to a scalar
    probabilities = torch.softmax(logits, dim=1)
    # cartport : probabailities.shape, logits.shape = [1,21]
    support = (
        torch.tensor([x for x in range(-support_size, support_size + 1)])
        .expand(probabilities.shape) # logit의 shape으로 바뀌었다.
        .float()
        .to(device=probabilities.device)
    )
    x = torch.sum(support * probabilities, dim=1, keepdim=True)
    # Invert the scaling (defined in https://arxiv.org/abs/1805.11593)
    # torch.sign(x) : 1 if x>0, -1 if x<0, 0 if x==0
    x = torch.sign(x) * (
        ((torch.sqrt(1 + 4 * 0.001 * (torch.abs(x) + 1 + 0.001)) - 1) / (2 * 0.001))
        ** 2
        - 1
    )
    # x.shape = [1,1]
    return x
