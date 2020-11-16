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


class DDPGLstmAttentionActor(nn.Module):
    pass

class DDPGLstmAttentionCritic(nn.Module):
    pass

class DDPGActor(nn.Module):
    def __init__(self, obs_size, hidden_size_1, hidden_size_2, n_actions, scale):
        super(DDPGActor, self).__init__()

        self.__name__ = "DDPGActor"

        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size_1),
            nn.ReLU(),
            nn.Linear(hidden_size_1, hidden_size_2),
            nn.ReLU(),
            nn.Linear(hidden_size_2, n_actions),
        )

        self.net.apply(init_weights)

        self.scale = scale

    def forward(self, x):
        n = self.net(x)
        t = torch.tanh(n)
        return t * self.scale


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


class D4PGCritic(nn.Module):
    def __init__(self, obs_size, hidden_size_1, hidden_size_2, n_actions, v_min, v_max, n_atoms):
        super(D4PGCritic, self).__init__()

        self.__name__ = "DDPGCritic"

        self.obs_net = nn.Sequential(
            nn.Linear(obs_size, hidden_size_1),
            nn.ReLU(),
        )

        self.out_net = nn.Sequential(
            nn.Linear(hidden_size_1 + n_actions, hidden_size_2),
            nn.ReLU(),
            nn.Linear(hidden_size_2, n_atoms)
        )

        delta = (v_max - v_min) / (n_atoms - 1)
        self.register_buffer("supports", torch.arange(v_min, v_max + delta, delta))

        self.obs_net.apply(init_weights)
        self.out_net.apply(init_weights)

    def forward(self, x, a):
        obs = self.obs_net(x)
        return self.out_net(torch.cat([obs, a], dim=1))

    def distribution_to_q_value(self, distribution):
        weights = F.softmax(distribution, dim=1) * self.supports
        res = weights.sum(dim=1)

        return res.unsqueeze(dim=-1)


def unpack_batch_for_policy_gradient(batch, net, params, device='cpu'):
    """
    Convert batch into training tensors
    :param batch:
    :param net:
    :return: states variable, actions tensor, target values variable
    """
    states, actions, rewards, not_done_idx, last_states = [], [], [], [], []

    for idx, exp in enumerate(batch):
        states.append(np.array(exp.state, copy=False))
        actions.append(int(exp.action))
        rewards.append(exp.reward)
        if exp.last_state is not None:
            not_done_idx.append(idx)
            last_states.append(np.array(exp.last_state, copy=False))

    states_v = torch.FloatTensor(np.array(states, copy=False)).to(device)
    actions_v = torch.LongTensor(actions).to(device)

    # handle rewards
    rewards_np = np.array(rewards, dtype=np.float32)

    if not_done_idx:
        last_states_v = torch.FloatTensor(np.array(last_states, copy=False)).to(device)
        last_values_v = net(last_states_v)[1]
        last_values_np = last_values_v.data.cpu().numpy()[:, 0] * params.GAMMA ** params.N_STEP
        rewards_np[not_done_idx] += last_values_np

    target_values_v = torch.FloatTensor(rewards_np).to(device)

    return states_v, actions_v, target_values_v


def unpack_batch_for_ddpg(batch, device="cpu"):
    states, actions, rewards, dones, last_states = [], [], [], [], []

    for exp in batch:
        states.append(np.array(exp.state, copy=False))
        actions.append(exp.action)
        rewards.append(exp.reward)
        dones.append(exp.last_state is None)
        if exp.last_state is None:
            last_states.append(exp.state)   # the result will be masked anyway
        else:
            last_states.append(np.array(exp.last_state, copy=False))

    states_v = rl_agent.float32_preprocessor(states).to(device)
    actions_v = rl_agent.float32_preprocessor(actions).to(device)
    rewards_v = rl_agent.float32_preprocessor(rewards).to(device)
    last_states_v = rl_agent.float32_preprocessor(last_states).to(device)
    dones_t = torch.BoolTensor(dones).to(device)
    return states_v, actions_v, rewards_v, dones_t, last_states_v


def distr_projection(next_distr_v, rewards_v, dones_mask_t, gamma, device="cpu"):
    next_distr = next_distr_v.data.cpu().numpy()
    rewards = rewards_v.data.cpu().numpy()
    dones_mask = dones_mask_t.cpu().numpy().astype(np.bool)
    batch_size = len(rewards)
    proj_distr = np.zeros((batch_size, N_ATOMS), dtype=np.float32)

    for atom in range(N_ATOMS):
        tz_j = np.minimum(Vmax, np.maximum(
            Vmin, rewards + (Vmin + atom * DELTA_Z) * gamma))
        b_j = (tz_j - Vmin) / DELTA_Z
        l = np.floor(b_j).astype(np.int64)
        u = np.ceil(b_j).astype(np.int64)
        eq_mask = u == l
        proj_distr[eq_mask, l[eq_mask]] += \
            next_distr[eq_mask, atom]
        ne_mask = u != l
        proj_distr[ne_mask, l[ne_mask]] += \
            next_distr[ne_mask, atom] * (u - b_j)[ne_mask]
        proj_distr[ne_mask, u[ne_mask]] += \
            next_distr[ne_mask, atom] * (b_j - l)[ne_mask]

    if dones_mask.any():
        proj_distr[dones_mask] = 0.0
        tz_j = np.minimum(Vmax, np.maximum(
            Vmin, rewards[dones_mask]))
        b_j = (tz_j - Vmin) / DELTA_Z
        l = np.floor(b_j).astype(np.int64)
        u = np.ceil(b_j).astype(np.int64)
        eq_mask = u == l
        eq_dones = dones_mask.copy()
        eq_dones[dones_mask] = eq_mask
        if eq_dones.any():
            proj_distr[eq_dones, l[eq_mask]] = 1.0
        ne_mask = u != l
        ne_dones = dones_mask.copy()
        ne_dones[dones_mask] = ne_mask
        if ne_dones.any():
            proj_distr[ne_dones, l[ne_mask]] = (u - b_j)[ne_mask]
            proj_distr[ne_dones, u[ne_mask]] = (b_j - l)[ne_mask]
    return torch.FloatTensor(proj_distr).to(device)