# https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F

from codes.c_models.advanced_exploration.noisy_net import NoisyLinear
from codes.c_models.discrete_action.discrete_action_model import DiscreteActionModel
from codes.e_utils.names import DeepLearningModelName


class DuelingDQNModel(DiscreteActionModel):
    def __init__(self, worker_id, observation_shape=None, action_n=None, params=None, device=None):
        super(DuelingDQNModel, self).__init__(worker_id, params, device)
        self.__name__ = "DuelingDQNModel"
        self.params = params

        if params.DEEP_LEARNING_MODEL == DeepLearningModelName.DUELING_DQN_MLP:
            num_inputs = observation_shape[0]
            self.base = DuelingDQN_MLP_Base(
                num_inputs=num_inputs, action_n=action_n, params=self.params
            )
        elif params.DEEP_LEARNING_MODEL == DeepLearningModelName.DUELING_DQN_CNN:
            self.base = DuelingDQN_CNN_Base(
                observation_shape=observation_shape, action_n=action_n, params=self.params
            )
        elif params.DEEP_LEARNING_MODEL == DeepLearningModelName.DUELING_DQN_SMALL_CNN:
            self.base = DuelingDQN_SmallCNN_Base(
                observation_shape=observation_shape, action_n=action_n, params=self.params
            )
        else:
            raise ValueError()

    def forward(self, inputs, agent_states=None):
        if not (type(inputs) is torch.Tensor):
            inputs = torch.tensor([inputs], dtype=torch.float).to(self.device)
        return self.base.forward(inputs)


class DuelingDQN_MLP_Base(nn.Module):
    def __init__(self, num_inputs, action_n, params):
        super(DuelingDQN_MLP_Base, self).__init__()
        self.__name__ = "DuelingDQN_MLP_Base"
        self.params = params

        self.action_n = action_n

        self.hidden_1_size = params.HIDDEN_1_SIZE
        self.hidden_2_size = params.HIDDEN_2_SIZE
        self.hidden_3_size = params.HIDDEN_3_SIZE

        self.net = nn.Sequential(
            nn.Linear(num_inputs, self.hidden_1_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_1_size, self.hidden_2_size),
            nn.LeakyReLU()
        )

        if self.params.NOISY_NET:
            self.noisy_value_1 = NoisyLinear(self.hidden_2_size, self.hidden_2_size)
            if self.params.DISTRIBUTIONAL:
                self.noisy_value_2 = NoisyLinear(self.hidden_3_size, self.params.NUM_SUPPORTS)
            else:
                self.noisy_value_2 = NoisyLinear(self.hidden_3_size, 1)

            self.noisy_advantage_1 = NoisyLinear(self.hidden_2_size, self.hidden_2_size)
            if self.params.DISTRIBUTIONAL:
                self.noisy_advantage_2 = NoisyLinear(self.hidden_3_size, action_n * self.params.NUM_SUPPORTS)
            else:
                self.noisy_advantage_2 = NoisyLinear(self.hidden_3_size, action_n)
        else:
            self.last_linear_value = nn.Linear(self.hidden_3_size, 1)
            self.last_linear_advantage = nn.Linear(self.hidden_3_size, action_n)

        self.fc_value = nn.Sequential(
            nn.Linear(self.hidden_2_size, self.hidden_3_size),
            nn.LeakyReLU()
        )

        self.fc_advantage = nn.Sequential(
            nn.Linear(self.hidden_2_size, self.hidden_3_size),
            nn.LeakyReLU()
        )

        self.layers_info = {
            "net": self.net,
            "fc_advantage": self.fc_advantage,
            "fc_value": self.fc_value
        }

        if self.params.NOISY_NET:
            self.layers_info["noisy_value_1"] = self.noisy_value_1
            self.layers_info["noisy_value_2"] = self.noisy_value_2
            self.layers_info["noisy_advantage_1"] = self.noisy_advantage_1
            self.layers_info["noisy_advantage_2"] = self.noisy_advantage_2
        else:
            self.layers_info["last_linear_advantage"] = self.last_linear_advantage
            self.layers_info["last_linear_value"] = self.last_linear_value

        self.train()

    @staticmethod
    def init_weights(m):
        if type(m) == nn.Linear:
            torch.nn.init.kaiming_normal_(m.weight)

    def forward(self, x):
        batch_size = x.size(0)

        if torch.is_tensor(x):
            x = x.to(torch.float32)
        else:
            x = torch.tensor(x, dtype=torch.float32)

        net_out = self.net(x)

        if self.params.NOISY_NET:
            net_out_value = self.noisy_value_1(net_out)
            value = self.fc_value(net_out_value)
            value = self.noisy_value_2(value)
            if self.params.DISTRIBUTIONAL:
                value = value.view(batch_size, 1, self.params.NUM_SUPPORTS)

            net_out_advantage = self.noisy_advantage_1(net_out)
            advantage = self.fc_advantage(net_out_advantage)
            advantage = self.noisy_advantage_2(advantage)
            if self.params.DISTRIBUTIONAL:
                advantage = advantage.view(batch_size, self.action_n, self.params.NUM_SUPPORTS)
        else:
            value = self.fc_value(net_out)
            value = self.last_linear_value(value)

            advantage = self.fc_advantage(net_out)
            advantage = self.last_linear_advantage(advantage)

        if self.params.DISTRIBUTIONAL:
            q_value = value + advantage - advantage.mean(1, keepdim=True)
            q_value = F.softmax(
                q_value.view(-1, self.params.NUM_SUPPORTS)
            ).view(-1, self.action_n, self.params.NUM_SUPPORTS)
        else:
            q_value = value + advantage - advantage.mean()
        return q_value

    def reset_noise(self):
        self.noisy_value_1.reset_noise()
        self.noisy_value_2.reset_noise()
        self.noisy_advantage_1.reset_noise()
        self.noisy_advantage_2.reset_noise()

    # def sample_noise(self):
    #     assert self.params.NOISY_NET
    #     for noisy_net in self.noisy_net_list:
    #         noisy_net.sample_noise()


class DuelingDQN_CNN_Base(nn.Module):
    def __init__(self, observation_shape, action_n, params):
        super(DuelingDQN_CNN_Base, self).__init__()
        self.__name__ = "DuelingDQN_CNN_Base"

        self.params = params
        self.action_n = action_n

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=observation_shape[0], out_channels=32, kernel_size=8, stride=4),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.LeakyReLU()
        )

        conv_out_size = self._get_conv_out(observation_shape)

        if self.params.NOISY_NET:
            self.noisy_value_1 = NoisyLinear(conv_out_size, conv_out_size)

            if self.params.DISTRIBUTIONAL:
                self.noisy_value_2 = NoisyLinear(512, self.params.NUM_SUPPORTS)
            else:
                self.noisy_value_2 = NoisyLinear(512, 1)

            self.noisy_advantage_1 = NoisyLinear(conv_out_size, conv_out_size)
            if self.params.DISTRIBUTIONAL:
                self.noisy_advantage_2 = NoisyLinear(512, action_n * self.params.NUM_SUPPORTS)
            else:
                self.noisy_advantage_2 = NoisyLinear(512, action_n)
        else:
            self.last_linear_value = nn.Linear(512, 1)
            self.last_linear_advantage = nn.Linear(512, action_n)

        self.fc_advantage = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.LeakyReLU()
        )
        self.fc_value = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.LeakyReLU()
        )

        # self.conv.apply(self.init_weights)
        # self.fc_adv.apply(self.init_weights)
        # self.fc_val.apply(self.init_weights)

        self.layers_info = {
            "conv": self.conv,
            "fc_advantage": self.fc_advantage,
            "fc_value": self.fc_value
        }

        if self.params.NOISY_NET:
            self.layers_info["noisy_value_1"] = self.noisy_value_1
            self.layers_info["noisy_value_2"] = self.noisy_value_2
            self.layers_info["noisy_advantage_1"] = self.noisy_advantage_1
            self.layers_info["noisy_advantage_2"] = self.noisy_advantage_2
        else:
            self.layers_info["last_linear_advantage"] = self.last_linear_advantage
            self.layers_info["last_linear_value"] = self.last_linear_value

        self.train()

    def init_weights(self, m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            torch.nn.init.kaiming_normal_(m.weight)

    def _get_conv_out(self, shape):
        o = self.conv(Variable(torch.zeros(1, *shape)))
        return int(np.prod(o.size()))

    def forward(self, x):
        batch_size = x.size(0)

        if torch.is_tensor(x):
            fx = x.to(torch.float32)
        else:
            fx = torch.tensor(x, dtype=torch.float32)

        conv_out = self.conv(fx).view(fx.size()[0], -1)

        if self.params.NOISY_NET:
            conv_out_value = self.noisy_value_1(conv_out)
            value = self.fc_value(conv_out_value)
            value = self.noisy_value_2(value)
            if self.params.DISTRIBUTIONAL:
                value = value.view(batch_size, 1, self.params.NUM_SUPPORTS)

            conv_out_advantage = self.noisy_advantage_1(conv_out)
            advantage = self.fc_advantage(conv_out_advantage)
            advantage = self.noisy_advantage_2(advantage)
            if self.params.DISTRIBUTIONAL:
                advantage = advantage.view(batch_size, self.action_n, self.params.NUM_SUPPORTS)
        else:
            value = self.fc_value(conv_out)
            value = self.last_linear_value(value)

            advantage = self.fc_advantage(conv_out)
            advantage = self.last_linear_advantage(advantage)

        return value + advantage - advantage.mean()

    def reset_noise(self):
        self.noisy_value_1.reset_noise()
        self.noisy_value_2.reset_noise()
        self.noisy_advantage_1.reset_noise()
        self.noisy_advantage_2.reset_noise()

    # def sample_noise(self):
    #     assert self.params.NOISY_NET
    #     for noisy_net in self.noisy_net_list:
    #         noisy_net.sample_noise()


class DuelingDQN_SmallCNN_Base(nn.Module):
    def __init__(self, observation_shape, action_n, params):
        super(DuelingDQN_SmallCNN_Base, self).__init__()

        self.__name__ = "DuelingDQN_SmallCNN_Base"

        self.params = params
        self.action_n = action_n

        self.conv = nn.Sequential(
            nn.Conv2d(observation_shape[0], 24, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(24, 32, kernel_size=2, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(32, 32, kernel_size=2, stride=1),
            nn.LeakyReLU()
        )

        conv_out_size = self._get_conv_out(observation_shape)

        if self.params.NOISY_NET:
            self.noisy_value_1 = NoisyLinear(conv_out_size, conv_out_size)
            if self.params.DISTRIBUTIONAL:
                self.noisy_value_2 = NoisyLinear(128, self.params.NUM_SUPPORTS)
            else:
                self.noisy_value_2 = NoisyLinear(128, 1)

            self.noisy_advantage_1 = NoisyLinear(conv_out_size, conv_out_size)
            if self.params.DISTRIBUTIONAL:
                self.noisy_advantage_2 = NoisyLinear(128, action_n * self.params.NUM_SUPPORTS)
            else:
                self.noisy_advantage_2 = NoisyLinear(128, action_n)
        else:
            self.last_linear_value = nn.Linear(128, 1)
            self.last_linear_advantage = nn.Linear(128, action_n)

        self.fc_advantage = nn.Sequential(
            nn.Linear(conv_out_size, 128),
            nn.LeakyReLU()
        )
        self.fc_value = nn.Sequential(
            nn.Linear(conv_out_size, 128),
            nn.LeakyReLU()
        )

        # self.conv.apply(self.init_weights)
        # self.fc_adv.apply(self.init_weights)
        # self.fc_val.apply(self.init_weights)

        self.layers_info = {
            "conv": self.conv,
            "fc_advantage": self.fc_advantage,
            "fc_value": self.fc_value
        }

        if self.params.NOISY_NET:
            self.layers_info["noisy_value_1"] = self.noisy_value_1
            self.layers_info["noisy_value_2"] = self.noisy_value_2
            self.layers_info["noisy_advantage_1"] = self.noisy_advantage_1
            self.layers_info["noisy_advantage_2"] = self.noisy_advantage_2
        else:
            self.layers_info["last_linear_advantage"] = self.last_linear_advantage
            self.layers_info["last_linear_value"] = self.last_linear_value

        self.train()

    def init_weights(self, m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            torch.nn.init.kaiming_normal_(m.weight)

    def _get_conv_out(self, shape):
        o = self.conv(Variable(torch.zeros(1, *shape)))
        return int(np.prod(o.size()))

    def forward(self, x):
        batch_size = x.size(0)

        if torch.is_tensor(x):
            fx = x.to(torch.float32)
        else:
            fx = torch.tensor(x, dtype=torch.float32)

        conv_out = self.conv(fx).view(fx.size()[0], -1)

        if self.params.NOISY_NET:
            conv_out_value = self.noisy_value_1(conv_out)
            value = self.fc_value(conv_out_value)
            value = self.noisy_value_2(value)
            if self.params.DISTRIBUTIONAL:
                value = value.view(batch_size, 1, self.params.NUM_SUPPORTS)

            conv_out_advantage = self.noisy_advantage_1(conv_out)
            advantage = self.fc_advantage(conv_out_advantage)
            advantage = self.noisy_advantage_2(advantage)
            if self.params.DISTRIBUTIONAL:
                advantage = advantage.view(batch_size, self.action_n, self.params.NUM_SUPPORTS)
        else:
            value = self.fc_value(conv_out)
            value = self.last_linear_value(value)

            advantage = self.fc_advantage(conv_out)
            advantage = self.last_linear_advantage(advantage)

        return value + advantage - advantage.mean()

    def reset_noise(self):
        self.noisy_value_1.reset_noise()
        self.noisy_value_2.reset_noise()
        self.noisy_advantage_1.reset_noise()
        self.noisy_advantage_2.reset_noise()

    # def sample_noise(self):
    #     assert self.params.NOISY_NET
    #     for noisy_net in self.noisy_net_list:
    #         noisy_net.sample_noise()