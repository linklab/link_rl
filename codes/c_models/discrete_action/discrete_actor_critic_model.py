# https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

from codes.c_models.discrete_action.discrete_action_model import DiscreteActionModel
from codes.e_utils.names import DeepLearningModelName


class DiscreteActorCriticModel(DiscreteActionModel):
    def __init__(self, worker_id, input_shape, num_outputs, params, device):
        super(DiscreteActorCriticModel, self).__init__(worker_id, params, device)
        self.__name__ = "DiscreteActorCriticModel"

        num_inputs = input_shape[0]

        if params.DEEP_LEARNING_MODEL == DeepLearningModelName.STOCHASTIC_DISCRETE_ACTOR_CRITIC_MLP:
            self.base = ActorCriticMLPBase(
                num_inputs=num_inputs, num_outputs=num_outputs, params=self.params
            )
        elif params.DEEP_LEARNING_MODEL == DeepLearningModelName.STOCHASTIC_DISCRETE_ACTOR_CRITIC_CNN:
            self.base = ActorCriticCNNBase(
                input_shape=input_shape, num_outputs=num_outputs
            )
        else:
            raise ValueError()

        self.reset_average_gradients()

    def forward(self, inputs):
        if not (type(inputs) is torch.Tensor):
            inputs = torch.tensor([inputs], dtype=torch.float).to(self.device)

        return self.base.forward(inputs)


class ActorCriticMLPBase(nn.Module):
    def __init__(self, num_inputs, num_outputs, params):
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
            nn.Linear(self.hidden_3_size, num_outputs)
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

    def forward(self, inputs):
        probs = self.forward_actor(inputs)
        critic_values = self.forward_critic(inputs)
        return probs, critic_values

    def forward_actor(self, inputs):
        x = self.actor(inputs)
        probs = F.softmax(x, dim=-1)
        return probs

    def forward_critic(self, inputs):
        critic_values = self.critic(inputs)
        return critic_values


class ActorCriticCNNBase(nn.Module):
    def __init__(self, input_shape, num_outputs):
        super(ActorCriticCNNBase, self).__init__()
        self.__name__ = "ActorCriticCNNBase"

        self.actor_conv = nn.Sequential(
            nn.Conv2d(in_channels=input_shape[0], out_channels=32, kernel_size=8, stride=4),
            nn.GELU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.GELU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.GELU()
        )

        self.critic_conv = nn.Sequential(
            nn.Conv2d(in_channels=input_shape[0], out_channels=32, kernel_size=8, stride=4),
            nn.GELU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.GELU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.GELU()
        )

        actor_conv_out_size = self._get_actor_conv_out(input_shape)
        critic_conv_out_size = self._get_critic_conv_out(input_shape)

        self.actor_fc = nn.Sequential(
            nn.Linear(actor_conv_out_size, 512),
            nn.LeakyReLU(),
            nn.Linear(512, num_outputs)
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

    def forward(self, inputs):
        probs = self.forward_actor(inputs)
        critic_values = self.forward_critic(inputs)
        return probs, critic_values

    def forward_actor(self, inputs):
        fx = inputs.float() / 256
        actor_conv_out = self.actor_conv(fx).view(fx.size()[0], -1)
        probs = F.softmax(self.actor_fc(actor_conv_out), dim=-1)
        return probs

    def forward_critic(self, inputs):
        fx = inputs.float() / 256
        critic_conv_out = self.critic_conv(fx).view(fx.size()[0], -1)
        critic_values = self.critic_fc(critic_conv_out)
        return critic_values
