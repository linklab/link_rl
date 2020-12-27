# https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail
import glob
import torch
import torch.nn as nn
import os
import torch.nn.functional as F

from common.fast_rl.value_based_model import DuelingDQNMLP
from common.models.base_model import BaseModel
from config.names import PROJECT_HOME, RLAlgorithmName


class DuelingDQNModel(BaseModel):
    def __init__(self, s_size, a_size, worker_id, params, device):
        super(DuelingDQNModel, self).__init__(s_size, a_size, worker_id, params, device)

        self.__name__ = "DuelingDQNModel"

        self.base = DuelingDQNMLP(
            obs_size=s_size,
            hidden_size_1=256,
            hidden_size_2=256,
            n_actions=a_size,
        )