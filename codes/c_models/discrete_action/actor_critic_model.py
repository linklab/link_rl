# https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail
import torch
import torch.nn as nn

from codes.c_models.base_model import BaseModel


class ActorCriticModel(BaseModel):
    def __init__(self, worker_id, num_inputs, num_outputs, params, device):
        super(ActorCriticModel, self).__init__(worker_id, params, device)
        self.__name__ = "ActorCriticModel"

        self.base = ActorCriticMLPBase(
            num_inputs=num_inputs, num_outputs=num_outputs, params=self.params
        )

        self.reset_average_gradients()

    def forward(self, inputs):
        if not (type(inputs) is torch.Tensor):
            inputs = torch.tensor([inputs], dtype=torch.float).to(self.device)
        return self.base.forward_actor(inputs), self.base.forward_critic(inputs)


class ActorCriticMLPBase(nn.Module):
    def __init__(self, num_inputs, num_outputs, params):
        super(ActorCriticMLPBase, self).__init__()
        self.__name__ = "ActorCriticMLPBase"
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

        #self.actor.apply(self.init_weights)

        self.critic = nn.Sequential(
            nn.Linear(num_inputs, self.hidden_1_size),
            nn.ReLU(),
            nn.Linear(self.hidden_1_size, self.hidden_2_size),
            nn.ReLU(),
            nn.Linear(self.hidden_2_size, self.hidden_3_size),
            nn.ReLU(),
            nn.Linear(self.hidden_3_size, 1),
        )

        #self.critic.apply(self.init_weights)

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

        return actions

    def forward_critic(self, inputs):
        critic_value = self.critic(inputs)

        return critic_value
