# https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

from codes.c_models.base_model import BaseModel
from codes.e_utils.names import DeepLearningModelName


class DiscreteActorCriticModel(BaseModel):
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
            nn.LeakyReLU(),
            nn.Linear(self.hidden_1_size, self.hidden_2_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_2_size, self.hidden_3_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_3_size, num_outputs)
        )

        self.actor.apply(self.init_weights)

        self.critic = nn.Sequential(
            nn.Linear(num_inputs, self.hidden_1_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_1_size, self.hidden_2_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_2_size, self.hidden_3_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_3_size, 1),
        )

        self.critic.apply(self.init_weights)

        self.actor_params = list(self.actor.parameters())
        self.critic_params = list(self.critic.parameters())

        self.layers_info = {'actor': self.actor, 'critic': self.critic}

        self.train()

    @staticmethod
    def init_weights(m):
        if type(m) == nn.Linear:
            torch.nn.init.kaiming_normal_(m.weight)

    def forward(self, inputs):
        actions = self.forward_actor(inputs)
        critic_values = self.forward_critic(inputs)
        return actions, critic_values

    def forward_actor(self, inputs):
        inputs = F.normalize(inputs)
        x = self.actor(inputs)
        actions = F.softmax(x, dim=-1)
        return actions

    def forward_critic(self, inputs):
        inputs = F.normalize(inputs)
        critic_values = self.critic(inputs)
        return critic_values


class ActorCriticCNNBase(nn.Module):
    def __init__(self, input_shape, num_outputs):
        super(ActorCriticCNNBase, self).__init__()
        self.__name__ = "ActorCriticCNNBase"

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=input_shape[0], out_channels=32, kernel_size=8, stride=4),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.LeakyReLU()
        )

        conv_out_size = self._get_conv_out(input_shape)

        self.actor_fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.LeakyReLU(),
            nn.Linear(512, num_outputs)
        )

        self.critic_fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 1)
        )

        self.conv.apply(self.init_weights)
        self.actor_fc.apply(self.init_weights)
        self.critic_fc.apply(self.init_weights)

        self.actor_params = list(self.conv.parameters()) + list(self.actor_fc.parameters())
        self.critic_params = list(self.critic_fc.parameters())

        self.layers_info = {'conv': self.conv, 'actor_fc': self.actor_fc, 'critic_fc': self.critic_fc}

        self.train()

    def _get_conv_out(self, shape):
        o = self.conv(Variable(torch.zeros(1, *shape)))
        return int(np.prod(o.size()))

    @staticmethod
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            torch.nn.init.kaiming_normal_(m.weight)
            # torch.nn.init.orthogonal(m.weight, gain=np.sqrt(2))

    def forward(self, inputs):
        inputs = F.normalize(inputs)
        actions = self.forward_actor(inputs)
        critic_values = self.forward_critic(inputs)
        return actions, critic_values

    def forward_actor(self, inputs):
        inputs = F.normalize(inputs)
        if torch.is_tensor(inputs):
            fx = inputs.to(torch.float32)
        else:
            fx = torch.tensor(inputs, dtype=torch.float32)

        conv_out = self.conv(fx).view(fx.size()[0], -1)
        actions = F.softmax(self.actor_fc(conv_out), dim=0)
        return actions

    def forward_critic(self, inputs):
        inputs = F.normalize(inputs)
        if torch.is_tensor(inputs):
            fx = inputs.to(torch.float32)
        else:
            fx = torch.tensor(inputs, dtype=torch.float32)

        conv_out = self.conv(fx).view(fx.size()[0], -1)
        critic_values = self.critic_fc(conv_out.detach())
        return critic_values




# class ActorCriticCNNBase(nn.Module):
#     def __init__(self, input_shape, num_outputs):
#         super(ActorCriticCNNBase, self).__init__()
#         self.__name__ = "ActorCriticCNNBase"
#
#         self.actor = ActorCNNBase(input_shape, num_outputs)
#         self.critic = CriticCNNBase(input_shape, 1)
#
#         self.layers_info = {'actor': self.actor, 'critic': self.critic}
#
#         self.train()
#
#     def forward(self, inputs):
#         return self.forward_actor(inputs)
#
#     def forward_actor(self, inputs):
#         actions = self.actor.forward_actor(inputs)
#         return actions
#
#     def forward_critic(self, inputs):
#         critic_values = self.critic.forward_critic(inputs)
#         return critic_values
#
#
# class ActorCNNBase(nn.Module):
#     def __init__(self, input_shape, num_outputs):
#         super(ActorCNNBase, self).__init__()
#         self.__name__ = "ActorCNNBase"
#
#         self.actor_conv = nn.Sequential(
#             nn.Conv2d(in_channels=input_shape[0], out_channels=32, kernel_size=8, stride=4),
#             nn.ReLU(),
#             nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
#             nn.ReLU(),
#             nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
#             nn.ReLU()
#         )
#
#         actor_conv_out_size = self._get_conv_out(self.actor_conv, input_shape)
#         self.actor_fc = nn.Sequential(
#             nn.Linear(actor_conv_out_size, 512),
#             nn.ReLU(),
#             nn.Linear(512, num_outputs)
#         )
#
#         # self.actor.apply(self.init_weights)
#
#         self.layers_info = {
#             'actor_conv': self.actor_conv, 'actor_fc': self.actor_fc,
#         }
#
#         self.train()
#
#     def _get_conv_out(self, conv, shape):
#         o = conv(Variable(torch.zeros(1, *shape)))
#         return int(np.prod(o.size()))
#
#     @staticmethod
#     def init_weights(m):
#         if type(m) == nn.Linear:
#             torch.nn.init.kaiming_normal_(m.weight)
#
#     def forward(self, inputs):
#         return self.forward_actor(inputs)
#
#     def forward_actor(self, inputs):
#         if torch.is_tensor(inputs):
#             fx = inputs.to(torch.float32)
#         else:
#             fx = torch.tensor(inputs, dtype=torch.float32)
#
#         actor_conv_out = self.actor_conv(fx).view(fx.size()[0], -1)
#         actions = self.actor_fc(actor_conv_out)
#         return actions
#
#
# class CriticCNNBase(nn.Module):
#     def __init__(self, input_shape, num_outputs=1):
#         super(CriticCNNBase, self).__init__()
#         self.__name__ = "CriticCNNBase"
#
#         self.critic_conv = nn.Sequential(
#             nn.Conv2d(in_channels=input_shape[0], out_channels=32, kernel_size=8, stride=4),
#             nn.ReLU(),
#             nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
#             nn.ReLU(),
#             nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
#             nn.ReLU()
#         )
#
#         critic_conv_out_size = self._get_conv_out(self.critic_conv, input_shape)
#         self.critic_fc = nn.Sequential(
#             nn.Linear(critic_conv_out_size, 512),
#             nn.ReLU(),
#             nn.Linear(512, num_outputs)
#         )
#
#         # self.critic.apply(self.init_weights)
#
#         self.layers_info = {
#             'critic_conv': self.critic_conv, 'critic_fc': self.critic_fc
#         }
#
#         self.train()
#
#     def _get_conv_out(self, conv, shape):
#         o = conv(Variable(torch.zeros(1, *shape)))
#         return int(np.prod(o.size()))
#
#     @staticmethod
#     def init_weights(m):
#         if type(m) == nn.Linear:
#             torch.nn.init.kaiming_normal_(m.weight)
#
#     def forward(self, inputs):
#         return self.forward_critic(inputs)
#
#     def forward_critic(self, inputs):
#         if torch.is_tensor(inputs):
#             fx = inputs.to(torch.float32)
#         else:
#             fx = torch.tensor(inputs, dtype=torch.float32)
#
#         critic_conv_out = self.critic_conv(fx).view(fx.size()[0], -1)
#         critic_values = self.critic_fc(critic_conv_out)
#
#         return critic_values
