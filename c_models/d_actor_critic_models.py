from collections import OrderedDict
from typing import Tuple
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Categorical

from c_models.a_models import Model
from g_utils.types import AgentMode


class ActorCritic(Model):
    def __init__(
            self, observation_shape: Tuple[int], n_out_actions: int, device=torch.device("cpu"), parameter=None
    ):
        super(ActorCritic, self).__init__(observation_shape, n_out_actions, device, parameter)

        if self.parameter.MODEL_TYPE == ModelType.LINEAR:
            input_n_features = self.observation_shape[0]
            self.fc_layers = self.get_linear_layers(input_n_features=input_n_features)
        elif self.parameter.MODEL_TYPE == ModelType.CONVOLUTIONAL:
            input_n_channels = self.observation_shape[0]
            self.conv_layers = self.get_conv_layers(input_n_channels=input_n_channels)
            conv_out_flat_size = self._get_conv_out(observation_shape)
            self.fc_layers = self.get_linear_layers(input_n_features=conv_out_flat_size)
        else:
            raise ValueError()

        self.fc_pi = nn.Linear(self.parameter.NEURONS_PER_FULLY_CONNECTED_LAYER[-1], self.n_out_actions)
        self.fc_v = nn.Linear(self.parameter.NEURONS_PER_FULLY_CONNECTED_LAYER[-1], 1)

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32, device=self.device)
        x = self.fc_layers(x)
        return x

    def pi(self, x):
        x = self.forward(x)
        x = self.fc_pi(x)
        prob = F.softmax(x, dim=-1)
        return prob

    def v(self, x):
        x = self.forward(x)
        v = self.fc_v(x)
        return v

    def get_action(self, x, mode=AgentMode.TRAIN):
        action_prob = self.pi(x)
        m = Categorical(probs=action_prob)
        if mode == AgentMode.TRAIN:
            action = m.sample()
        else:
            action = torch.argmax(m.probs, dim=1 if action_prob.dim() == 2 else 0)
        return action.cpu().numpy()


class ContinuousActorCritic(ActorCritic):
    def __init__(
            self, observation_shape: Tuple[int], n_out_actions: int, device=torch.device("cpu"), parameter=None
    ):
        super(ContinuousActorCritic, self).__init__(observation_shape, n_out_actions, device, parameter)

        num_inputs = observation_shape[0]

        if params.DEEP_LEARNING_MODEL == DeepLearningModelName.CONTINUOUS_STOCHASTIC_ACTOR_CRITIC_MLP:
            self.base = StochasticActorCriticMLPBase(
                num_inputs=num_inputs, num_outputs=num_outputs, params=self.params
            )
            self.__name__ = "StochasticContinuousActorCriticMLPModel"
        elif params.DEEP_LEARNING_MODEL == DeepLearningModelName.CONTINUOUS_STOCHASTIC_ACTOR_CRITIC_CNN:
            self.base = None
            self.__name__ = "StochasticContinuousActorCriticCNNModel"
        elif params.DEEP_LEARNING_MODEL == DeepLearningModelName.CONTINUOUS_STOCHASTIC_ACTOR_CRITIC_RNN:
            self.base = None
            self.__name__ = "StochasticContinuousActorCriticRNNModel"
        else:
            raise ValueError()

    def forward(self, inputs, agent_state):
        if isinstance(self.base, RNNModel):
            if isinstance(agent_state, list):
                actor_hidden_states, critic_hidden_states = [], []
                for each_agent_state in agent_state:
                    actor_hidden_states.append(each_agent_state.actor_hidden_state)
                    critic_hidden_states.append(each_agent_state.critic_hidden_state)
                actor_hidden_states_v = float32_preprocessor(actor_hidden_states).to(self.device)
                critic_hidden_states_v = float32_preprocessor(critic_hidden_states).to(self.device)
            else:
                actor_hidden_states_v = agent_state.actor_hidden_state
                critic_hidden_states_v = agent_state.critic_hidden_state

            mu, logstd, new_actor_hidden_state = self.base.forward_actor(inputs, actor_hidden_states_v)
            value, new_critic_hidden_state = self.base.forward_critic(inputs, critic_hidden_states_v)

            agent_state = rl_utils.initial_agent_state(
                actor_hidden_state=new_actor_hidden_state, critic_hidden_state=new_critic_hidden_state
            )
        else:
            mu, logstd = self.base.forward_actor(inputs)
            value = self.base.forward_critic(inputs)

        return mu, logstd, value, agent_state

    def forward_actor(self, inputs, actor_hidden_state):
        if isinstance(self.base, RNNModel):
            mu, logstd = self.base.forward_actor(inputs)
        else:
            mu, logstd = self.base.forward_actor(inputs)
        return mu, logstd, actor_hidden_state

    def forward_critic(self, inputs, critic_hidden_state):
        if isinstance(self.base, RNNModel):
            values = self.base.forward_citic(inputs)
        else:
            values = self.base.forward_critic(inputs)
        return values, critic_hidden_state
