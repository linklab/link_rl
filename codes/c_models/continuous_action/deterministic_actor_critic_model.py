# https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail
import torch
import torch.nn as nn
import torch.nn.functional as F

from codes.c_models.base_model import BaseModel
from codes.e_utils.names import RLAlgorithmName


class DeterministicActorCriticModel(BaseModel):
    def __init__(self, worker_id, input_shape, num_outputs, params, device):
        super(DeterministicActorCriticModel, self).__init__(worker_id, params, device)
        self.__name__ = "DeterministicActorCriticModel"

        num_inputs = input_shape[0]

        if self.params.RL_ALGORITHM == RLAlgorithmName.DDPG_V0:
            self.base = DeterministicActorCriticMLPBase(
                num_inputs=num_inputs, num_outputs=num_outputs, params=self.params
            )
        elif self.params.RL_ALGORITHM == RLAlgorithmName.D4PG_V0:
            self.base = DistributionalActorCriticMLPBase(
                num_inputs=num_inputs, num_outputs=num_outputs, params=self.params
            )
        else:
            raise ValueError()

        self.reset_average_gradients()


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
            nn.ReLU(),
            nn.Linear(self.hidden_1_size, self.hidden_2_size),
            nn.ReLU(),
            nn.Linear(self.hidden_2_size, self.hidden_3_size),
            nn.ReLU(),
            nn.Linear(self.hidden_3_size, num_outputs),
        )

        self.actor.apply(self.init_weights)

        self.critic = nn.Sequential(
            nn.Linear(num_inputs + num_outputs, self.hidden_1_size),
            nn.ReLU(),
            nn.Linear(self.hidden_1_size, self.hidden_2_size),
            nn.ReLU(),
            nn.Linear(self.hidden_2_size, self.hidden_3_size),
            nn.ReLU(),
            nn.Linear(self.hidden_3_size, 1),
        )

        self.critic.apply(self.init_weights)

        self.layers_info = {'actor': self.actor, 'critic': self.critic}

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

        return actions * self.params.ACTION_SCALE

    def forward_critic(self, inputs, actions):
        critic_value = self.critic(torch.cat([inputs, actions], dim=-1))

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
