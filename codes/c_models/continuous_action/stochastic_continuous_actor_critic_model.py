import torch
import torch.nn as nn

from codes.c_models.base_model import RNNModel
from codes.c_models.continuous_action.continuous_action_model import ContinuousActionModel
from codes.d_agents.a0_base_agent import float32_preprocessor
from codes.e_utils import rl_utils
from codes.e_utils.names import DeepLearningModelName


class StochasticContinuousActorCriticModel(ContinuousActionModel):
    def __init__(self, worker_id, input_shape, num_outputs, params, device):
        super(StochasticContinuousActorCriticModel, self).__init__(worker_id, params, device)

        num_inputs = input_shape[0]

        if params.DEEP_LEARNING_MODEL == DeepLearningModelName.STOCHASTIC_CONTINUOUS_ACTOR_CRITIC_MLP:
            self.base = StochasticActorCriticMLPBase(
                num_inputs=num_inputs, num_outputs=num_outputs, params=self.params
            )
            self.__name__ = "StochasticContinuousActorCriticMLPModel"
        elif params.DEEP_LEARNING_MODEL == DeepLearningModelName.STOCHASTIC_CONTINUOUS_ACTOR_CRITIC_CNN:
            self.base = None
            self.__name__ = "StochasticContinuousActorCriticCNNModel"
        elif params.DEEP_LEARNING_MODEL == DeepLearningModelName.STOCHASTIC_CONTINUOUS_ACTOR_CRITIC_RNN:
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


class StochasticActorCriticMLPBase(nn.Module):
    def __init__(self, num_inputs, num_outputs, params):
        super(StochasticActorCriticMLPBase, self).__init__()
        self.__name__ = "StochasticActorCriticMLPBase"
        self.params = params

        self.hidden_1_size = params.HIDDEN_1_SIZE
        self.hidden_2_size = params.HIDDEN_2_SIZE
        self.hidden_3_size = params.HIDDEN_3_SIZE

        self.actor = ActorMLPBase(num_inputs, num_outputs, params)

        self.critic = nn.Sequential(
            nn.Linear(num_inputs, self.hidden_1_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_1_size, self.hidden_2_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_2_size, self.hidden_3_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_3_size, 1),
        )

        self.layers_info = {'actor': self.actor, 'critic': self.critic}

        self.actor_params = list(self.actor.parameters())
        self.critic_params = list(self.critic.parameters())

        self.train()

    def forward(self, inputs):
        mu, logstd = self.forward_actor(inputs)
        value = self.forward_critic(inputs)
        return mu, logstd, value

    def forward_actor(self, inputs):
        mu, logstd = self.actor(inputs)
        return mu, logstd

    def forward_critic(self, inputs):
        return self.critic(inputs)


class ActorMLPBase(nn.Module):
    def __init__(self, num_inputs, num_outputs, params):
        super(ActorMLPBase, self).__init__()
        self.__name__ = "ActorMLPBase"
        self.params = params

        self.hidden_1_size = params.HIDDEN_1_SIZE
        self.hidden_2_size = params.HIDDEN_2_SIZE
        self.hidden_3_size = params.HIDDEN_3_SIZE

        self.common = nn.Sequential(
            nn.Linear(num_inputs, self.hidden_1_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_1_size, self.hidden_2_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_2_size, self.hidden_3_size),
            nn.LeakyReLU()
        )

        self.mu = nn.Sequential(
            nn.Linear(self.hidden_3_size, num_outputs),
            nn.Tanh()
        )

        self.logstd = nn.Sequential(
            nn.Linear(self.hidden_3_size, num_outputs),
            nn.Softplus()
        )

    @staticmethod
    def init_weights(m):
        if type(m) == nn.Linear:
            torch.nn.init.kaiming_normal_(m.weight)

    def forward(self, inputs):
        # if inputs.size()[0] == 1:
        #     print(inputs[0][2], inputs[0][5], inputs[0][5], "!!!!!!!!!!!!!!!!1")
        #self.check_nan_parameters()

        mu_v = self.mu(self.common(inputs))
        logstd_v = self.logstd(self.common(inputs))

        if torch.isnan(mu_v[0][0]):
            print("inputs:", inputs, "!!! - 1")
            print("self.common(inputs)", self.common(inputs), "!!! - 2")
            print("mu_v:", mu_v, "!!! - 3")
            print("logstd_v:", logstd_v, "!!! - 4")
            exit(-1)

        return mu_v, logstd_v

    def check_nan_parameters(self):
        for param in self.mu.parameters():
            print(param.data[0])
        #     if (param.data != param.data).any():
        #         print(param.data)
        #
        # for param in self.mu.parameters():
        #     if (param.data != param.data).any():
        #         print(param.data)
        #
        # for param in self.logstd.parameters():
        #     if (param.data != param.data).any():
        #         print(param.data)
