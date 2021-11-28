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
            self, n_features=4, n_actions=2, device=torch.device("cpu"), params=None
    ):
        super(QNet, self).__init__()
        self.n_features = n_features
        self.n_actions = n_actions
        self.device = device
        self.params = params

        fc_layers_dict = OrderedDict()
        fc_layers_dict["fc_0"] = nn.Linear(n_features, self.params.NEURONS_PER_LAYER[0])
        fc_layers_dict["fc_0_activation"] = nn.LeakyReLU()

        for idx in range(1, len(self.params.NEURONS_PER_LAYER) - 1):
            fc_layers_dict["fc_{0}".format(idx)] = nn.Linear(
                self.params.NEURONS_PER_LAYER[idx], self.params.NEURONS_PER_LAYER[idx + 1]
            )
            fc_layers_dict["fc_{0}_activation".format(idx)] = nn.LeakyReLU()

        self.fc_layers = nn.Sequential(fc_layers_dict)
        self.fc_last = nn.Linear(self.params.NEURONS_PER_LAYER[-1], n_actions)

        self.version = 0

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32, device=self.device)

        x = self.fc_layers(x)
        x = self.fc_last(x)

        return x


class AtariCNN(nn.Module):
    def __init__(
            self, obs_shape: Tuple[int], n_actions: int, hidden_size: int = 256,
            device=torch.device("cpu"), params=None
    ):
        super(AtariCNN, self).__init__()

        input_channel = obs_shape[0]

        self.conv = nn.Sequential(
            nn.Conv2d(input_channel, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(obs_shape)

        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions)
        )

        self.device = device
        self.params = params

    def _get_conv_out(self, shape):
        cont_out = self.conv(torch.zeros(1, *shape))
        return int(np.prod(cont_out.size()))

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32, device=self.device)

        conv_out = self.conv(x)

        conv_out = torch.flatten(conv_out, start_dim=1)
        out = self.fc(conv_out)
        return out

    def get_action(self, observation, epsilon):
        if random.random() < epsilon:
            action = random.randint(0, 2)
            return action
        else:
            # Convert to Tensor
            observation = np.array(observation, copy=False)
            observation = torch.tensor(observation, device=self.device)

            # Add batch-dim
            if len(observation.shape) == 3:
                observation = observation.unsqueeze(dim=0)

            q_values = self.forward(observation)
            action = torch.argmax(q_values, dim=1)
            return action.cpu().numpy()


class Policy(nn.Module):
    def __init__(
            self, n_features=4, n_actions=2, device=torch.device("cpu"), params=None
    ):
        super(Policy, self).__init__()
        self.device = device
        self.params = params

        fc_layers_dict = OrderedDict()
        fc_layers_dict["fc_0"] = nn.Linear(n_features, self.params.NEURONS_PER_LAYER[0])
        fc_layers_dict["fc_0_activation"] = nn.LeakyReLU()

        for idx in range(1, len(self.params.NEURONS_PER_LAYER) - 1):
            fc_layers_dict["fc_{0}".format(idx)] = nn.Linear(
                self.params.NEURONS_PER_LAYER[idx], self.params.NEURONS_PER_LAYER[idx + 1]
            )
            fc_layers_dict["fc_{0}_activation".format(idx)] = nn.LeakyReLU()

        self.fc_layers = nn.Sequential(fc_layers_dict)
        self.fc_last = nn.Linear(self.params.NEURONS_PER_LAYER[-1], n_actions)

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
            self, n_features=4, n_actions=2, device=torch.device("cpu"), params=None
    ):
        super(ActorCritic, self).__init__()
        self.device = device
        self.params = params

        fc_layers_dict = OrderedDict()
        fc_layers_dict["fc_0"] = nn.Linear(n_features, self.params.NEURONS_PER_LAYER[0])
        fc_layers_dict["fc_0_activation"] = nn.LeakyReLU()

        for idx in range(1, len(self.params.NEURONS_PER_LAYER) - 1):
            fc_layers_dict["fc_{0}".format(idx)] = nn.Linear(
                self.params.NEURONS_PER_LAYER[idx], self.params.NEURONS_PER_LAYER[idx + 1]
            )
            fc_layers_dict["fc_{0}_activation".format(idx)] = nn.LeakyReLU()

        self.fc_layers = nn.Sequential(fc_layers_dict)

        self.fc_pi = nn.Linear(self.params.NEURONS_PER_LAYER[-1], n_actions)
        self.fc_v = nn.Linear(self.params.NEURONS_PER_LAYER[-1], 1)

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
