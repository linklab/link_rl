import glob
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np

from common.fast_rl import rl_agent


def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        torch.nn.init.kaiming_normal_(m.weight)


class A2CMLP(nn.Module):
    def __init__(self, obs_size, hidden_size_1, hidden_size_2, n_actions):
        super(A2CMLP, self).__init__()

        self.__name__ = "A2CMLP"

        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size_1),
            nn.ReLU(),
            nn.Linear(hidden_size_1, hidden_size_2),
            nn.ReLU()
        )

        self.policy = nn.Sequential(
            nn.Linear(hidden_size_2, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )

        self.value = nn.Sequential(
            nn.Linear(hidden_size_2, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def init_weights(self, m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            torch.nn.init.kaiming_normal_(m.weight)

    def forward(self, x):
        if torch.is_tensor(x):
            x = x.to(torch.float32)
        else:
            x = torch.tensor(x, dtype=torch.float32)
        net_out = self.net(x)
        policy = self.policy(net_out)
        value = self.value(net_out)

        return policy, value


class ContinuousA2CMLP(nn.Module):
    def __init__(self, obs_size, hidden_size_1, hidden_size_2, n_actions):
        super(ContinuousA2CMLP, self).__init__()

        self.__name__ = "ContinuousA2CMLP"

        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size_1),
            nn.ReLU(),
            nn.Linear(hidden_size_1, hidden_size_2),
            nn.ReLU()
        )

        self.mu = nn.Sequential(
            nn.Linear(hidden_size_2, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions),
            nn.Tanh()
        )

        self.var = nn.Sequential(
            nn.Linear(hidden_size_2, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions),
            nn.Softplus(),
        )

        self.value = nn.Sequential(
            nn.Linear(hidden_size_2, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        net_out = self.net(x)
        return self.mu(net_out), self.var(net_out), self.value(net_out)


class DDPGActor(nn.Module):
    def __init__(self, obs_size, hidden_size_1, hidden_size_2, n_actions):
        super(DDPGActor, self).__init__()

        self.__name__ = "DDPGActor"

        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size_1),
            nn.ReLU(),
            nn.Linear(hidden_size_1, hidden_size_2),
            nn.ReLU(),
            nn.Linear(hidden_size_2, n_actions),
            nn.Tanh()
        )

        self.net.apply(init_weights)

    def forward(self, x):
        return self.net(x)


class DDPGCritic(nn.Module):
    def __init__(self, obs_size, hidden_size_1, hidden_size_2, n_actions):
        super(DDPGCritic, self).__init__()

        self.__name__ = "DDPGCritic"

        self.obs_net = nn.Sequential(
            nn.Linear(obs_size, hidden_size_1),
            nn.ReLU(),
        )

        self.out_net = nn.Sequential(
            nn.Linear(hidden_size_1 + n_actions, hidden_size_2),
            nn.ReLU(),
            nn.Linear(hidden_size_2, 1)
        )

        self.obs_net.apply(init_weights)
        self.out_net.apply(init_weights)

    def forward(self, x, a):
        obs = self.obs_net(x)
        return self.out_net(torch.cat([obs, a], dim=1))


def unpack_batch_for_policy_gradient(batch, net, params, device='cpu'):
    """
    Convert batch into training tensors
    :param batch:
    :param net:
    :return: states variable, actions tensor, target values variable
    """
    batch_states = []
    batch_actions = []
    batch_rewards = []
    batch_not_done_idx = []
    batch_last_states = []
    for idx, exp in enumerate(batch):
        batch_states.append(np.array(exp.state, copy=False))
        batch_actions.append(int(exp.action))
        batch_rewards.append(exp.reward)
        if exp.last_state is not None:
            batch_not_done_idx.append(idx)
            batch_last_states.append(np.array(exp.last_state, copy=False))

    batch_states_v = torch.FloatTensor(np.array(batch_states, copy=False)).to(device)
    batch_actions_v = torch.LongTensor(batch_actions).to(device)

    # handle rewards
    batch_rewards_np = np.array(batch_rewards, dtype=np.float32)

    if batch_not_done_idx:
        batch_last_states_v = torch.FloatTensor(np.array(batch_last_states, copy=False)).to(device)
        batch_last_values_v = net(batch_last_states_v)[1]
        batch_last_values_np = batch_last_values_v.data.cpu().numpy()[:, 0]
        batch_last_values_np *= params.GAMMA ** params.N_STEP
        batch_rewards_np[batch_not_done_idx] += batch_last_values_np

    batch_target_values_v = torch.FloatTensor(batch_rewards_np).to(device)

    return batch_states_v, batch_actions_v, batch_target_values_v


def unpack_batch_for_ddpg(batch, device="cpu"):
    states, actions, rewards, dones, last_states = [], [], [], [], []
    for exp in batch:
        states.append(exp.state)
        actions.append(exp.action)
        rewards.append(exp.reward)
        dones.append(exp.last_state is None)
        if exp.last_state is None:
            last_states.append(exp.state)
        else:
            last_states.append(exp.last_state)
    states_v = rl_agent.float32_preprocessor(states).to(device)
    actions_v = rl_agent.float32_preprocessor(actions).to(device)
    rewards_v = rl_agent.float32_preprocessor(rewards).to(device)
    last_states_v = rl_agent.float32_preprocessor(last_states).to(device)
    dones_t = torch.BoolTensor(dones).to(device)
    return states_v, actions_v, rewards_v, dones_t, last_states_v