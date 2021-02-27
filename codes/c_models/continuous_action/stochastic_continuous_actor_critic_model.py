import torch
import torch.nn as nn

from codes.c_models.base_model import BaseModel


class StochasticContinuousActorCriticModel(BaseModel):
    def __init__(self, worker_id, input_shape, num_outputs, params, device):
        super(StochasticContinuousActorCriticModel, self).__init__(worker_id, params, device)
        self.__name__ = "StochasticContinuousActorCriticModel"

        num_inputs = input_shape[0]

        self.base = StochasticActorCriticMLPBase(
            num_inputs=num_inputs, num_outputs=num_outputs, params=self.params
        )

        self.reset_average_gradients()

    def forward(self, inputs):
        if not (type(inputs) is torch.Tensor):
            inputs = torch.tensor([inputs], dtype=torch.float).to(self.device)

        mu, var, value = self.base.forward(inputs)

        return mu, var, value


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

        self.train()

    def forward(self, inputs):
        mu, var = self.actor(inputs)
        value = self.critic(inputs)
        return mu, var, value

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

        self.net = nn.Sequential(
            nn.Linear(num_inputs, self.hidden_1_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_1_size, self.hidden_2_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_2_size, self.hidden_3_size),
            nn.LeakyReLU(),
        )

        self.mu = nn.Sequential(
            nn.Linear(self.hidden_3_size, num_outputs),
            nn.Tanh()
        )

        self.var = nn.Sequential(
            nn.Linear(self.hidden_3_size, num_outputs),
            nn.Softplus()
        )

    @staticmethod
    def init_weights(m):
        if type(m) == nn.Linear:
            torch.nn.init.kaiming_normal_(m.weight)

    def forward(self, inputs):
        net_out = self.net(inputs)
        mu = 2 * self.mu(net_out)
        var = self.var(net_out) + 1e-3
        return mu, var
