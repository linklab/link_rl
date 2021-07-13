# https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

from codes.c_models.base_model import RNNModel
from codes.c_models.discrete_action.discrete_action_model import DiscreteActionModel
from codes.e_utils import rl_utils
from codes.e_utils.common_utils import float32_preprocessor
from codes.e_utils.names import DeepLearningModelName


class DiscreteActorCriticModel(DiscreteActionModel):
    def __init__(self, worker_id, observation_shape, action_n, params, device):
        super(DiscreteActorCriticModel, self).__init__(worker_id, params, device)

        num_inputs = observation_shape[0]

        if params.DEEP_LEARNING_MODEL == DeepLearningModelName.DISCRETE_STOCHASTIC_ACTOR_CRITIC_MLP:
            self.base = ActorCriticMLPBase(num_inputs=num_inputs, action_n=action_n, params=self.params)
            self.__name__ = "DiscreteActorCriticMLPModel"
        elif params.DEEP_LEARNING_MODEL == DeepLearningModelName.DISCRETE_STOCHASTIC_ACTOR_CRITIC_CNN:
            self.base = ActorCriticCNNBase(observation_shape=observation_shape, action_n=action_n)
            self.__name__ = "DiscreteActorCriticCNNModel"
        elif params.DEEP_LEARNING_MODEL == DeepLearningModelName.DISCRETE_STOCHASTIC_ACTOR_CRITIC_RNN:
            self.base = ActorCriticRNNBase(num_inputs=num_inputs, action_n=action_n, params=self.params)
            self.__name__ = "DiscreteActorCriticRNNModel"
        else:
            raise ValueError()

    # def forward(self, input, agent_state=None):
    #     if not (type(input) is torch.Tensor):
    #         input = torch.tensor([input], dtype=torch.float).to(self.device)
    #
    #     return self.base.forward(input, agent_state)

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

            probs, new_actor_hidden_state = self.base.forward_actor(inputs, actor_hidden_states_v)
            value, new_critic_hidden_state = self.base.forward_critic(inputs, critic_hidden_states_v)

            agent_state = rl_utils.initial_agent_state(
                actor_hidden_state=new_actor_hidden_state, critic_hidden_state=new_critic_hidden_state
            )
        else:
            probs = self.base.forward_actor(inputs)
            value = self.base.forward_critic(inputs)

        return probs, value, agent_state

    def forward_actor(self, inputs, actor_hidden_state):
        if isinstance(self.base, RNNModel):
            probs, actor_hidden_state = self.base.forward_actor(inputs, actor_hidden_state=actor_hidden_state)
        else:
            probs = self.base.forward_actor(inputs)
        return probs, actor_hidden_state

    def forward_critic(self, inputs, critic_hidden_state):
        if isinstance(self.base, RNNModel):
            values = self.base.forward_citic(inputs)
        else:
            values = self.base.forward_critic(inputs)
        return values, critic_hidden_state


class ActorCriticMLPBase(nn.Module):
    def __init__(self, num_inputs, action_n, params):
        super(ActorCriticMLPBase, self).__init__()
        self.__name__ = "ActorCriticMLPBase"
        self.params = params

        self.hidden_1_size = params.HIDDEN_1_SIZE
        self.hidden_2_size = params.HIDDEN_2_SIZE
        self.hidden_3_size = params.HIDDEN_3_SIZE

        #self.common.apply(self.init_weights)

        self.actor = nn.Sequential(
            nn.Linear(num_inputs, self.hidden_1_size),
            nn.GELU(),
            nn.Linear(self.hidden_1_size, self.hidden_2_size),
            nn.GELU(),
            nn.Linear(self.hidden_2_size, self.hidden_3_size),
            nn.GELU(),
            nn.Linear(self.hidden_3_size, action_n)
        )

        # self.actor.apply(self.init_weights)

        self.critic = nn.Sequential(
            nn.Linear(num_inputs, self.hidden_1_size),
            nn.GELU(),
            nn.Linear(self.hidden_1_size, self.hidden_2_size),
            nn.GELU(),
            nn.Linear(self.hidden_2_size, self.hidden_3_size),
            nn.GELU(),
            nn.Linear(self.hidden_3_size, 1),
        )

        # self.critic.apply(self.init_weights)

        self.actor_params = list(self.actor.parameters())
        self.critic_params = list(self.critic.parameters())

        self.layers_info = {'actor': self.actor, 'critic': self.critic}

        self.train()

    # @staticmethod
    # def init_weights(m):
    #     if type(m) == nn.Linear:
    #         torch.nn.init.kaiming_normal_(m.weight)

    def forward(self, input):
        probs = self.forward_actor(input)
        critic_values = self.forward_critic(input)
        return probs, critic_values

    def forward_actor(self, input):
        x = self.actor(input)
        probs = F.softmax(x, dim=-1)
        return probs

    def forward_critic(self, input):
        critic_values = self.critic(input)
        return critic_values


class ActorCriticRNNBase(RNNModel):
    def __init__(self, num_inputs, action_n, params):
        super(ActorCriticRNNBase, self).__init__()
        self.__name__ = "ActorCriticRNNBase"
        self.params = params

        self.num_directions = 2 if params.RNN_BIDIRECTIONAL else 1

        self.actor_rnn = nn.GRU(
            input_size=num_inputs,
            hidden_size=params.RNN_HIDDEN_SIZE,
            num_layers=params.RNN_NUM_LAYER,
            bias=True, batch_first=True,
            bidirectional=params.RNN_BIDIRECTIONAL
        )
        self.actor_linear = nn.Linear(
            in_features=params.RNN_HIDDEN_SIZE * self.num_directions, out_features=action_n
        )

        self.critic_rnn = nn.LSTM(
            input_size=num_inputs,
            hidden_size=params.RNN_HIDDEN_SIZE,
            num_layers=params.RNN_NUM_LAYER,
            bias=True, batch_first=True,
            bidirectional=params.RNN_BIDIRECTIONAL
        )
        self.critic_linear = nn.Linear(
            in_features=params.RNN_HIDDEN_SIZE * self.num_directions, out_features=1
        )

        self.actor_params = list(self.actor_rnn.parameters()) + list(self.actor_linear.parameters())
        self.critic_params = list(self.critic_rnn.parameters()) + list(self.critic_linear.parameters())

        self.layers_info = {
            'actor_rnn': self.actor_rnn, 'actor_linear': self.actor_linear,
            'critic_rnn': self.critic_rnn, 'critic_linear': self.critic_linear
        }

        self.train()

    # @staticmethod
    # def init_weights(m):
    #     if type(m) == nn.Linear:
    #         torch.nn.init.kaiming_normal_(m.weight)

    def forward(self, input, agent_state):
        probs, new_actor_hidden_state = self.forward_actor(input, agent_state.actor_hidden_state)
        critic_values, new_critic_hidden_state = self.forward_critic(input, agent_state.critic_hidden_state)

        new_agent_state = [new_actor_hidden_state, new_critic_hidden_state]

        return probs, critic_values, new_agent_state

    def forward_actor(self, input, actor_hidden_state):
        input = torch.unsqueeze(input, dim=1)
        actor_output, new_actor_hidden_state = self.actor_rnn(input, actor_hidden_state)
        probs = self.actor_linear(actor_output)

        return probs, new_actor_hidden_state

    def forward_critic(self, input, critic_hidden_state):
        critic_output, new_critic_hidden_state = self.critic_rnn(input, critic_hidden_state)
        critic_values = self.critic_linear(critic_output)

        return critic_values, new_critic_hidden_state


class ActorCriticCNNBase(nn.Module):
    def __init__(self, observation_shape, action_n):
        super(ActorCriticCNNBase, self).__init__()
        self.__name__ = "ActorCriticCNNBase"

        self.actor_conv = nn.Sequential(
            nn.Conv2d(in_channels=observation_shape[0], out_channels=32, kernel_size=(8, 8), stride=(4, 4)),
            nn.GELU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(4, 4), stride=(2, 2)),
            nn.GELU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1)),
            nn.GELU()
        )

        self.critic_conv = nn.Sequential(
            nn.Conv2d(in_channels=observation_shape[0], out_channels=32, kernel_size=(8, 8), stride=(4, 4)),
            nn.GELU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(4, 4), stride=(2, 2)),
            nn.GELU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1)),
            nn.GELU()
        )

        actor_conv_out_size = self._get_actor_conv_out(observation_shape)
        critic_conv_out_size = self._get_critic_conv_out(observation_shape)

        self.actor_fc = nn.Sequential(
            nn.Linear(actor_conv_out_size, 512),
            nn.LeakyReLU(),
            nn.Linear(512, action_n)
        )

        self.critic_fc = nn.Sequential(
            nn.Linear(critic_conv_out_size, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 1)
        )

        # self.conv.apply(self.init_weights)
        # self.actor_fc.apply(self.init_weights)
        # self.critic_fc.apply(self.init_weights)

        self.actor_params = list(self.actor_conv.parameters()) + list(self.actor_fc.parameters())
        self.critic_params = list(self.critic_conv.parameters()) + list(self.critic_fc.parameters())

        self.layers_info = {'actor_conv': self.actor_conv, 'critic_conv': self.critic_conv, 'actor_fc': self.actor_fc, 'critic_fc': self.critic_fc}

        self.train()

    def _get_actor_conv_out(self, shape):
        o = self.actor_conv(Variable(torch.zeros(1, *shape)))
        return int(np.prod(o.size()))

    def _get_critic_conv_out(self, shape):
        o = self.critic_conv(Variable(torch.zeros(1, *shape)))
        return int(np.prod(o.size()))

    # @staticmethod
    # def init_weights(m):
    #     if type(m) == nn.Linear or type(m) == nn.Conv2d:
    #         torch.nn.init.kaiming_normal_(m.weight)
    #         # torch.nn.init.orthogonal(m.weight, gain=np.sqrt(2))

    def forward(self, input):
        probs, _ = self.forward_actor(input)
        critic_values, _ = self.forward_critic(input)
        return probs, critic_values

    def forward_actor(self, input):
        fx = input.float() / 256
        actor_conv_out = self.actor_conv(fx).view(fx.size()[0], -1)
        probs = F.softmax(self.actor_fc(actor_conv_out), dim=-1)
        return probs

    def forward_critic(self, input):
        fx = input.float() / 256
        critic_conv_out = self.critic_conv(fx).view(fx.size()[0], -1)
        critic_values = self.critic_fc(critic_conv_out)
        return critic_values
