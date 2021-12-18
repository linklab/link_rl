import random
from collections import OrderedDict
from typing import Tuple
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Categorical

from g_utils.types import AgentMode


class Policy(nn.Module):
    def __init__(
            self, observation_shape: Tuple[int], n_out_actions: int, device=torch.device("cpu"), parameter=None
    ):
        super(Policy, self).__init__()
        self.device = device
        self.parameter = parameter

        fc_layers_dict = OrderedDict()
        fc_layers_dict["fc_0"] = nn.Linear(observation_shape[0], self.parameter.NEURONS_PER_LAYER[0])
        fc_layers_dict["fc_0_activation"] = nn.LeakyReLU()

        for idx in range(1, len(self.parameter.NEURONS_PER_LAYER) - 1):
            fc_layers_dict["fc_{0}".format(idx)] = nn.Linear(
                self.parameter.NEURONS_PER_LAYER[idx], self.parameter.NEURONS_PER_LAYER[idx + 1]
            )
            fc_layers_dict["fc_{0}_activation".format(idx)] = nn.LeakyReLU()

        self.fc_layers = nn.Sequential(fc_layers_dict)
        self.fc_last = nn.Linear(self.parameter.NEURONS_PER_LAYER[-1], n_out_actions)

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

    # def get_action_with_action_prob_selected(self, x, mode=AgentMode.TRAIN):
    #     action_prob = self.forward(x)   # [0.3, 0.7]
    #     m = Categorical(probs=action_prob)
    #
    #     if mode == AgentMode.TRAIN:
    #         action = m.sample()
    #         action_prob_selected = action_prob[action]
    #     else:
    #         action = torch.argmax(m.probs, dim=-1)
    #         action_prob_selected = None
    #     return action.cpu().numpy(), action_prob_selected
