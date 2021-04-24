# https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail
import torch
import torch.nn as nn
import torch.nn.functional as F

from codes.c_models.advanced_exploration.noisy_net import NoisyLinear
from codes.c_models.base_model import BaseModel
from codes.e_utils.names import RLAlgorithmName


class DeterministicContinuousActorCriticModel(BaseModel):
    def __init__(self, worker_id, input_shape, num_outputs, params, device):
        super(DeterministicContinuousActorCriticModel, self).__init__(worker_id, params, device)
        self.__name__ = "DeterministicContinuousActorCriticModel"

        num_inputs = input_shape[0]

        if self.params.RL_ALGORITHM == RLAlgorithmName.DDPG_V0:
            self.base = DeterministicActorCriticMLPBase(
                num_inputs=num_inputs, num_outputs=num_outputs, params=self.params
            )
        elif self.params.RL_ALGORITHM == RLAlgorithmName.D4PG_V0:
            self.base = DistributionalActorCriticMLPBase(
                num_inputs=num_inputs, num_outputs=num_outputs, params=self.params
            )
        elif self.params.RL_ALGORITHM == RLAlgorithmName.TD3_V0:
            self.base = DeterministicActorCriticTD3MLPBase(
                num_inputs=num_inputs, num_outputs=num_outputs, params=self.params
            )
        else:
            raise ValueError()

        self.reset_average_gradients()

    def forward_actor(self, inputs):
        return self.base.forward_actor(inputs)

    def forward_critic(self, inputs, actions):
        return self.base.forward_critic(inputs, actions)


class DeterministicActorCriticMLPBase(nn.Module):
    def __init__(self, num_inputs, num_outputs, params):
        super(DeterministicActorCriticMLPBase, self).__init__()
        self.__name__ = "DeterministicActorCriticMLPBase"
        self.params = params

        self.hidden_1_size = params.HIDDEN_1_SIZE
        self.hidden_2_size = params.HIDDEN_2_SIZE
        self.hidden_3_size = params.HIDDEN_3_SIZE

        self.actor = nn.Sequential(
            nn.Linear(num_inputs, self.hidden_1_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_1_size, self.hidden_2_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_2_size, self.hidden_3_size),
            nn.LeakyReLU()
        )

        if self.params.NOISY_NET:
            self.last_actor = nn.Sequential(
                NoisyLinear(self.hidden_3_size, self.hidden_3_size),
                nn.Linear(self.hidden_3_size, num_outputs)
            )
        else:
            self.last_actor = nn.Linear(self.hidden_3_size, num_outputs)

        # self.actor.apply(self.init_weights)

        self.critic = nn.Sequential(
            nn.Linear(num_inputs + num_outputs, self.hidden_1_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_1_size, self.hidden_2_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_2_size, self.hidden_3_size),
            nn.LeakyReLU()
        )

        if self.params.NOISY_NET:
            self.last_critic = nn.Sequential(
                NoisyLinear(self.hidden_3_size, self.hidden_3_size),
                nn.Linear(self.hidden_3_size, 1)
            )
        else:
            self.last_critic = nn.Linear(self.hidden_3_size, 1)

        # self.critic.apply(self.init_weights)

        self.actor_params = list(self.actor.parameters())
        self.critic_params = list(self.critic.parameters())

        self.layers_info = {'actor': self.actor, 'critic': self.critic}

        self.train()

    @staticmethod
    def init_weights(m):
        if type(m) == nn.Linear:
            torch.nn.init.kaiming_normal_(m.weight)

    def forward(self, inputs):
        return self.forward_actor(inputs)

    def forward_actor(self, inputs):
        outs = self.actor(inputs)
        actions = self.last_actor(outs)
        actions = torch.tanh(actions)
        return actions

    def forward_critic(self, inputs, actions):
        outs = self.critic(torch.cat([inputs, actions], dim=-1))
        critic_value = self.last_critic(outs)
        return critic_value


class DistributionalActorCriticMLPBase(DeterministicActorCriticMLPBase):
    def __init__(self, num_inputs, num_outputs, params):
        super(DistributionalActorCriticMLPBase, self).__init__(num_inputs, num_outputs, params)
        self.__name__ = "DistributionalActorCriticMLPBase"

        self.logstd = nn.Parameter(torch.zeros(num_outputs))

        delta = (params.V_MAX - params.V_MIN) / (params.N_ATOMS - 1)
        self.register_buffer("supports", torch.arange(params.V_MIN, params.V_MAX + delta, delta))

    def distribution_to_q(self, distribution):
        weights = F.softmax(distribution, dim=-1) * self.supports
        res = weights.sum(dim=-1)
        return res.unsqueeze(dim=-1)


class DeterministicActorCriticTD3MLPBase(nn.Module):
    def __init__(self, num_inputs, num_outputs, params):
        super(DeterministicActorCriticTD3MLPBase, self).__init__()
        self.__name__ = "DeterministicActorCriticTD3MLPBase"
        self.params = params

        self.num_inputs = num_inputs
        self.num_outputs = num_outputs

        self.hidden_1_size = params.HIDDEN_1_SIZE
        self.hidden_2_size = params.HIDDEN_2_SIZE
        self.hidden_3_size = params.HIDDEN_3_SIZE

        self.actor = nn.Sequential(
            nn.Linear(num_inputs, self.hidden_1_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_1_size, self.hidden_2_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_2_size, self.hidden_3_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_3_size, num_outputs)
        )

        # self.actor.apply(self.init_weights)

        self.critic_1 = nn.Sequential(
            nn.Linear(num_inputs + num_outputs, self.hidden_1_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_1_size, self.hidden_2_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_2_size, self.hidden_3_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_3_size, 1)
        )

        self.critic_2 = nn.Sequential(
            nn.Linear(num_inputs + num_outputs, self.hidden_1_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_1_size, self.hidden_2_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_2_size, self.hidden_3_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_3_size, 1)
        )

        # self.critic.apply(self.init_weights)

        self.critic_params = list(self.critic_1.parameters()) + list(self.critic_2.parameters())
        # self.critic_1_params = list(self.critic_1.parameters())
        # self.critic_2_params = list(self.critic_2.parameters())

        self.layers_info = {'actor': self.actor, 'critic_1': self.critic_1, 'critic_2': self.critic_2}

        self.train()

    @staticmethod
    def init_weights(m):
        if type(m) == nn.Linear:
            torch.nn.init.kaiming_normal_(m.weight)

    def forward(self, inputs):
        return self.forward_actor(inputs)

    def forward_actor(self, inputs):
        actions = self.actor(inputs)
        actions = torch.tanh(actions)

        return actions

    def forward_critic(self, inputs, actions):
        critic_1_value = self.critic_1(torch.cat([inputs, actions], dim=-1))
        critic_2_value = self.critic_2(torch.cat([inputs, actions], dim=-1))

        return critic_1_value, critic_2_value

    def forward_only_critic_1(self, inputs, actions):
        critic_1_value = self.critic_1(torch.cat([inputs, actions], dim=-1))

        return critic_1_value
