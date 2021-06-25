import numpy as np
import torch
import torch.nn as nn

from codes.c_models.continuous_action.continuous_action_model import ContinuousActionModel


class StochasticContinuousActorCriticModel(ContinuousActionModel):
    def __init__(self, worker_id, input_shape, num_outputs, params, device):
        super(StochasticContinuousActorCriticModel, self).__init__(worker_id, params, device)
        self.__name__ = "StochasticContinuousActorCriticModel"

        num_inputs = input_shape[0]

        self.base = StochasticActorCriticMLPBase(
            num_inputs=num_inputs, num_outputs=num_outputs, params=self.params, device=self.device
        )

        self.reset_average_gradients()

    def forward(self, inputs):
        if not (type(inputs) is torch.Tensor):
            inputs = torch.tensor([inputs], dtype=torch.float).to(self.device)

        mu, value = self.base.forward(inputs)

        return mu, value


class StochasticActorCriticMLPBase(nn.Module):
    def __init__(self, num_inputs, num_outputs, params, device):
        super(StochasticActorCriticMLPBase, self).__init__()
        self.__name__ = "StochasticActorCriticMLPBase"
        self.params = params
        self.device = device

        self.hidden_1_size = params.HIDDEN_1_SIZE
        self.hidden_2_size = params.HIDDEN_2_SIZE
        self.hidden_3_size = params.HIDDEN_3_SIZE

        self.actor = ActorMLPBase(num_inputs, num_outputs, params, device)

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
        mu = self.actor(inputs)
        value = self.critic(inputs)
        return mu, value

    def forward_critic(self, inputs):
        return self.critic(inputs)


class ActorMLPBase(nn.Module):
    def __init__(self, num_inputs, num_outputs, params, device):
        super(ActorMLPBase, self).__init__()
        self.__name__ = "ActorMLPBase"
        self.params = params
        self.device = device
        self.num_outputs = num_outputs

        self.hidden_1_size = params.HIDDEN_1_SIZE
        self.hidden_2_size = params.HIDDEN_2_SIZE
        self.hidden_3_size = params.HIDDEN_3_SIZE

        self.mu = nn.Sequential(
            nn.Linear(num_inputs, self.hidden_1_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_1_size, self.hidden_2_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_2_size, self.hidden_3_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_3_size, self.num_outputs),
            nn.Tanh()
        )

        self.action_variance = None

        self.set_action_variance(action_std=0.1)

    @staticmethod
    def init_weights(m):
        if type(m) == nn.Linear:
            torch.nn.init.kaiming_normal_(m.weight)

    def set_action_variance(self, action_std):
        self.action_variance = torch.full(size=(self.num_outputs,), fill_value=action_std * action_std).to(self.device)

    def forward(self, inputs):
        # if inputs.size()[0] == 1:
        #     print(inputs[0][2], inputs[0][5], inputs[0][5], "!!!!!!!!!!!!!!!!1")
        #self.check_nan_parameters()

        mu_v = self.mu(inputs)

        if torch.isnan(mu_v[0][0]):
            print("inputs:", inputs, "!!! - 1")
            print("self.common(inputs)", self.mu(inputs), "!!! - 2")
            print("mu_v:", mu_v, "!!! - 3")
            exit(-1)

        return mu_v

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

