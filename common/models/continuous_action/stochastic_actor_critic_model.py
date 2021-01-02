# https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.models.base_model import BaseModel
from config.names import RLAlgorithmName


class StochasticActorCriticModel(BaseModel):
    def __init__(self, s_size, a_size, worker_id, params, device):
        super(StochasticActorCriticModel, self).__init__(s_size, a_size, worker_id, params, device)
        self.__name__ = "DeterministicActorCriticModel"

        self.base = StochasticActorCriticMLPBase(
            num_inputs=s_size,
            num_ouputs=a_size,
            params=self.params
        )

        self.reset_average_gradients()

    def forward(self, inputs):
        if not (type(inputs) is torch.Tensor):
            inputs = torch.tensor([inputs], dtype=torch.float).to(self.device)
        return self.base.forward(inputs)

    def act(self, inputs):
        raise NotImplementedError()


class StochasticActorCriticMLPBase(nn.Module):
    def __init__(self, num_inputs, num_ouputs, params):
        super(StochasticActorCriticMLPBase, self).__init__()
        self.__name__ = "StochasticActorCriticMLPBase"
        self.params = params

        self.hidden_1_size = params.HIDDEN_1_SIZE
        self.hidden_2_size = params.HIDDEN_2_SIZE
        self.hidden_3_size = params.HIDDEN_3_SIZE

        self.net = nn.Sequential(
            nn.Linear(num_inputs, self.hidden_1_size),
            nn.ReLU(),
            nn.Linear(self.hidden_1_size, self.hidden_2_size),
            nn.ReLU()
        )

        self.mu = nn.Sequential(
            nn.Linear(self.hidden_2_size, self.hidden_3_size),
            nn.ReLU(),
            nn.Linear(self.hidden_3_size, num_ouputs),
            nn.Tanh()
        )

        self.var = nn.Sequential(
            nn.Linear(self.hidden_2_size, self.hidden_3_size),
            nn.ReLU(),
            nn.Linear(self.hidden_3_size, num_ouputs),
            nn.Softplus(),
        )

        self.net.apply(self.init_weights)
        self.mu.apply(self.init_weights)
        self.var.apply(self.init_weights)

        self.value = nn.Sequential(
            nn.Linear(self.hidden_2_size, self.hidden_3_size),
            nn.ReLU(),
            nn.Linear(self.hidden_3_size, 1),
        )

        self.value.apply(self.init_weights)

        self.layers_info = {'net': self.net, 'mu': self.mu, 'var': self.var, 'value': self.value}

        self.train()

    @staticmethod
    def init_weights(m):
        if type(m) == nn.Linear:
            torch.nn.init.kaiming_normal_(m.weight)

    def forward(self, inputs):
        net_out = self.net(inputs)
        return self.mu(net_out), self.var(net_out), self.value(net_out)

    def forward_critic(self, inputs):
        net_out = self.net(inputs)
        value = self.value(net_out)
        return value
