import random
from collections import OrderedDict
from typing import Tuple
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Categorical

from g_utils.types import AgentMode


class ActorCritic(nn.Module):
    def __init__(
            self, observation_shape: Tuple[int], n_out_actions: int, device=torch.device("cpu"), parameter=None
    ):
        super(ActorCritic, self).__init__()
        self.device = device
        self.parameter = parameter

        fc_layers_dict = OrderedDict()
        fc_layers_dict["fc_0"] = nn.Linear(observation_shape[0], self.parameter.NEURONS_PER_LAYER[0])
        fc_layers_dict["fc_0_activation"] = nn.LeakyReLU()

        for idx in range(1, len(self.parameter.NEURONS_PER_LAYER) - 1):
            fc_layers_dict["fc_{0}".format(idx)] = nn.Linear(
                self.parameter.NEURONS_PER_LAYER[idx], self.parameter.NEURONS_PER_LAYER[idx + 1]
            )
            fc_layers_dict["fc_{0}_activation".format(idx)] = nn.LeakyReLU()

        self.fc_layers = nn.Sequential(fc_layers_dict)

        self.fc_pi = nn.Linear(self.parameter.NEURONS_PER_LAYER[-1], n_actions)
        self.fc_v = nn.Linear(self.parameter.NEURONS_PER_LAYER[-1], 1)

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
