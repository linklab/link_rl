# https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail
import torch
import torch.nn as nn

from codes.c_models.base_model import BaseModel


class StochasticActorCriticModel(BaseModel):
    def __init__(self, worker_id, input_shape, num_outputs, params, device):
        super(StochasticActorCriticModel, self).__init__(worker_id, params, device)
        self.__name__ = "StochasticActorCriticModel"

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

        self.common = nn.Sequential(
            nn.Linear(num_inputs, self.hidden_1_size),
            nn.ReLU(),
            nn.Linear(self.hidden_1_size, self.hidden_2_size),
            nn.ReLU(),
            nn.Linear(self.hidden_2_size, self.hidden_3_size),
            nn.ReLU(),
        )

        self.actor_mu = nn.Sequential(
            nn.Linear(self.hidden_3_size, num_outputs),
            nn.Tanh()
        )

        self.actor_var = nn.Sequential(
            nn.Linear(self.hidden_3_size, num_outputs),
            nn.Softplus()
        )

        self.critic = nn.Sequential(
            nn.Linear(self.hidden_3_size, self.hidden_3_size),
            nn.ReLU(),
            nn.Linear(self.hidden_3_size, 1)
        )

        self.layers_info = {
            'common': self.common, 'actor_mu': self.actor_mu, 'actor_var': self.actor_var, 'critic': self.critic
        }

        self.train()

    def forward(self, inputs):
        common_out = self.common(inputs)
        mu = self.actor_mu(common_out)
        var = self.actor_var(common_out.detach())
        value = self.critic(common_out.detach())
        return mu, var, value

    def forward_actor(self, inputs):
        common_out = self.common(inputs)
        mu = self.actor_mu(common_out)
        var = self.actor_var(common_out.detach())
        return mu, var

    def forward_critic(self, inputs):
        common_out = self.common(inputs.detach())
        return self.critic(common_out)
