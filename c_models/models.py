import random
from collections import OrderedDict
from typing import Tuple
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Categorical

from g_utils.types import AgentMode


class QNet(nn.Module):
    def __init__(
            self, n_features: int, n_actions: int, device=torch.device("cpu"), parameter=None
    ):
        super(QNet, self).__init__()
        self.n_features = n_features
        self.n_actions = n_actions
        self.device = device
        self.parameter = parameter

        fc_layers_dict = OrderedDict()
        fc_layers_dict["fc_0"] = nn.Linear(n_features, self.parameter.NEURONS_PER_LAYER[0])
        fc_layers_dict["fc_0_activation"] = nn.LeakyReLU()

        for idx in range(1, len(self.parameter.NEURONS_PER_LAYER) - 1):
            fc_layers_dict["fc_{0}".format(idx)] = nn.Linear(
                self.parameter.NEURONS_PER_LAYER[idx], self.parameter.NEURONS_PER_LAYER[idx + 1]
            )
            fc_layers_dict["fc_{0}_activation".format(idx)] = nn.LeakyReLU()

        self.fc_layers = nn.Sequential(fc_layers_dict)
        self.fc_last = nn.Linear(self.parameter.NEURONS_PER_LAYER[-1], n_actions)

        self.version = 0

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32, device=self.device)

        x = self.fc_layers(x)
        x = self.fc_last(x)

        return x


class CnnQNet(nn.Module):
    def __init__(
            self, obs_shape: Tuple[int], n_actions: int, device=torch.device("cpu"), parameter=None
    ):
        super(CnnQNet, self).__init__()
        self.obs_shape = obs_shape
        self.n_actions = n_actions
        self.device = device
        self.parameter = parameter

        input_channel = obs_shape[0]

        conv_layers_dict = OrderedDict()
        conv_layers_dict["conv_0"] = nn.Conv2d(
            in_channels=input_channel,
            out_channels=self.parameter.OUT_CHANNELS_PER_LAYER[0],
            kernel_size=self.parameter.KERNEL_SIZE_PER_LAYER[0],
            stride=self.parameter.STRIDE_PER_LAYER[0]
        )
        conv_layers_dict["conv_0_activation"] = nn.LeakyReLU()

        for idx in range(1, len(self.parameter.OUT_CHANNELS_PER_LAYER)):
            conv_layers_dict["conv_{0}".format(idx)] = nn.Conv2d(
                in_channels=self.parameter.OUT_CHANNELS_PER_LAYER[idx-1],
                out_channels=self.parameter.OUT_CHANNELS_PER_LAYER[idx],
                kernel_size=self.parameter.KERNEL_SIZE_PER_LAYER[idx],
                stride=self.parameter.STRIDE_PER_LAYER[idx]
            )
            conv_layers_dict["conv_{0}_activation".format(idx)] = nn.LeakyReLU()

        self.conv_layers = nn.Sequential(conv_layers_dict)
        conv_out_flat_size = self._get_conv_out(obs_shape)

        fc_layers_dict = OrderedDict()
        fc_layers_dict["fc_0"] = nn.Linear(
            conv_out_flat_size, self.parameter.NEURONS_PER_FULLY_CONNECTED_LAYER[0]
        )
        fc_layers_dict["fc_0_activation"] = nn.LeakyReLU()

        if len(self.parameter.NEURONS_PER_FULLY_CONNECTED_LAYER) >= 2:
            for idx in range(1, len(self.parameter.NEURONS_PER_FULLY_CONNECTED_LAYER) - 1):
                fc_layers_dict["fc_{0}".format(idx)] = nn.Linear(
                    self.parameter.NEURONS_PER_FULLY_CONNECTED_LAYER[idx],
                    self.parameter.NEURONS_PER_FULLY_CONNECTED_LAYER[idx + 1]
                )
                fc_layers_dict["fc_{0}_activation".format(idx)] = nn.LeakyReLU()

        self.fc_layers = nn.Sequential(fc_layers_dict)
        self.fc_last = nn.Linear(
            self.parameter.NEURONS_PER_FULLY_CONNECTED_LAYER[-1], n_actions
        )

    def _get_conv_out(self, shape):
        cont_out = self.conv_layers(torch.zeros(1, *shape))
        return int(np.prod(cont_out.size()))

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32, device=self.device)

        conv_out = self.conv_layers(x)
        conv_out = torch.flatten(conv_out, start_dim=1)
        out = self.fc_layers(conv_out)
        out = self.fc_last(out)
        return out

    # def get_action(self, observation, epsilon):
    #     if random.random() < epsilon:
    #         action = random.randint(0, 2)
    #         return action
    #     else:
    #         # Convert to Tensor
    #         observation = np.array(observation, copy=False)
    #         observation = torch.tensor(observation, device=self.device)
    #
    #         # Add batch-dim
    #         if len(observation.shape) == 3:
    #             observation = observation.unsqueeze(dim=0)
    #
    #         q_values = self.forward(observation)
    #         action = torch.argmax(q_values, dim=1)
    #         return action.cpu().numpy()


class Policy(nn.Module):
    def __init__(
            self, obs_shape: Tuple[int], n_actions: int, device=torch.device("cpu"), parameter=None
    ):
        super(Policy, self).__init__()
        self.device = device
        self.parameter = parameter

        fc_layers_dict = OrderedDict()
        fc_layers_dict["fc_0"] = nn.Linear(obs_shape[0], self.parameter.NEURONS_PER_LAYER[0])
        fc_layers_dict["fc_0_activation"] = nn.LeakyReLU()

        for idx in range(1, len(self.parameter.NEURONS_PER_LAYER) - 1):
            fc_layers_dict["fc_{0}".format(idx)] = nn.Linear(
                self.parameter.NEURONS_PER_LAYER[idx], self.parameter.NEURONS_PER_LAYER[idx + 1]
            )
            fc_layers_dict["fc_{0}_activation".format(idx)] = nn.LeakyReLU()

        self.fc_layers = nn.Sequential(fc_layers_dict)
        self.fc_last = nn.Linear(self.parameter.NEURONS_PER_LAYER[-1], n_actions)

        # self.fc1 = nn.Linear(n_features, 128)
        # self.fc2 = nn.Linear(128, 128)
        # self.fc3 = nn.Linear(128, n_actions)

    def forward(self, x):
        # x = [1.0, 0.5, 0.8, 0.8]  --> [1.7, 2.3] --> [0.3, 0.7]
        # x = [
        #  [1.0, 0.5, 0.8, 0.8]
        #  [1.0, 0.5, 0.8, 0.8]
        #  [1.0, 0.5, 0.8, 0.8]
        #  ...
        #  [1.0, 0.5, 0.8, 0.8]
        # ]

        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32, device=self.device)

        x = self.fc_layers(x)
        x = self.fc_last(x)
        x = F.softmax(x, dim=-1)

        return x

    def get_action(self, x, mode=AgentMode.TRAIN):
        action_prob = self.forward(x)
        m = Categorical(probs=action_prob)

        if mode == AgentMode.TRAIN:
            action = m.sample()
        else:
            action = torch.argmax(m.probs, dim=-1)
        return action.cpu().numpy()

    def get_action_with_action_prob_selected(self, x, mode=AgentMode.TRAIN):
        action_prob = self.forward(x)   # [0.3, 0.7]
        m = Categorical(probs=action_prob)

        if mode == AgentMode.TRAIN:
            action = m.sample()
            action_prob_selected = action_prob[action]
        else:
            action = torch.argmax(m.probs, dim=-1)
            action_prob_selected = None
        return action.cpu().numpy(), action_prob_selected


class ActorCritic(nn.Module):
    def __init__(
            self, obs_shape: Tuple[int], n_actions: int, device=torch.device("cpu"), parameter=None
    ):
        super(ActorCritic, self).__init__()
        self.device = device
        self.parameter = parameter

        fc_layers_dict = OrderedDict()
        fc_layers_dict["fc_0"] = nn.Linear(obs_shape[0], self.parameter.NEURONS_PER_LAYER[0])
        fc_layers_dict["fc_0_activation"] = nn.LeakyReLU()

        for idx in range(1, len(self.parameter.NEURONS_PER_LAYER) - 1):
            fc_layers_dict["fc_{0}".format(idx)] = nn.Linear(
                self.parameter.NEURONS_PER_LAYER[idx], self.parameter.NEURONS_PER_LAYER[idx + 1]
            )
            fc_layers_dict["fc_{0}_activation".format(idx)] = nn.LeakyReLU()

        self.fc_layers = nn.Sequential(fc_layers_dict)

        self.fc_pi = nn.Linear(self.parameter.NEURONS_PER_LAYER[-1], n_actions)
        self.fc_v = nn.Linear(self.parameter.NEURONS_PER_LAYER[-1], 1)

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32, device=self.device)
        x = self.fc_layers(x)
        return x

    def pi(self, x):
        x = self.forward(x)
        x = self.fc_pi(x)
        prob = F.softmax(x, dim=-1)
        return prob

    def v(self, x):
        x = self.forward(x)
        v = self.fc_v(x)
        return v

    def get_action(self, x, mode=AgentMode.TRAIN):
        action_prob = self.pi(x)
        m = Categorical(probs=action_prob)
        if mode == AgentMode.TRAIN:
            action = m.sample()
        else:
            action = torch.argmax(m.probs, dim=1 if action_prob.dim() == 2 else 0)
        return action.cpu().numpy()
