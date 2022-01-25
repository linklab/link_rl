# https://github.com/pranz24/pytorch-soft-actor-critic/blob/master/sac.py
# https://github.com/BY571/Soft-Actor-Critic-and-Extensions/blob/master/SAC.py
# PAPER: https://arxiv.org/abs/1812.05905
# https://www.pair.toronto.edu/csc2621-w20/assets/slides/lec4_sac.pdf
# https://bair.berkeley.edu/blog/2017/10/06/soft-q-learning/
import torch.optim as optim
import numpy as np
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
from gym.spaces import Discrete, Box
from torch.distributions import Categorical, Normal

from c_models.g_sac_models import ContinuousSacModel, DiscreteSacModel
from d_agents.agent import Agent
from g_utils.types import AgentMode

# 저장되는 transition이 다르다.
# get action에 num_simulation만큼 반복하는 mcts 구현
#
class AgentMuZero(Agent):
    def __init__(self, observation_space, action_space, parameter):
        super(AgentMuZero, self).__init__(observation_space, action_space, parameter)

        if isinstance(self.action_space, Discrete):
            pass
        elif isinstance(self.action_space, Box):
            pass
        else:
            raise ValueError()

    def get_action(self, obs, mode=AgentMode.TRAIN):
        # MCTS 구
        if isinstance(self.action_space, Discrete):
            pass
        elif isinstance(self.action_space, Box):
            pass

    def train_muzero(self, training_steps_v):
        if isinstance(self.action_space, Discrete):
            # next_actions_v = None
            # next_log_prob_v = None
            pass
        elif isinstance(self.action_space, Box):
            pass
        else:
            raise ValueError()


class SelfPlay:
    pass


class MCTS:
    def __init__(self):
        pass

    def run(self):
        pass

    def select_child(self):
        pass

    def ucb_score(self):
        pass

    def backpropagate(self):
        pass


class Node:
    def __init__(self):
        pass

    def value(self):
        pass

    def expanded(self):
        pass

    def add_exploration_noise(self):
        pass
