import glob
import math
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


#####################################
## DDPGLstmAttention: Begin        ##
#####################################
class GruEncoder(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, n_layers=1, dropout=0., bidirectional=True):
        super(GruEncoder, self).__init__()
        self.bidirectional = bidirectional

        self.rnn = nn.GRU(
            embedding_dim, hidden_dim, n_layers,
            batch_first=True,
            dropout=dropout, bidirectional=bidirectional
        )

    def forward(self, input, hidden=None):
        return self.rnn(input, hidden)


class NullEmbedding(nn.Module):
    def __init__(self):
        super(NullEmbedding, self).__init__()

    def forward(self, input):
        return input


class Attention(nn.Module):
    def __init__(self, query_dim, key_dim, value_dim):
        super(Attention, self).__init__()
        self.scale = 1. / math.sqrt(query_dim) # Scaled Dot Product

    def forward(self, query, keys, values):
        # Query = [BxH]       B: Batch Size, Q: Hidden Size
        # Keys = [BxSxH]      B: Batch Size, S: Step Length, H: Hidden Size
        # Values = [BxSxH]    B: Batch Size, S: Step Length, H: Hidden Size
        # Outputs = score:[BxS], attention_value:[BxH]

        # Here we assume q_dim == k_dim (dot product attention)

        query = query.unsqueeze(1)                  # [BxH] -> [Bx1xH]
        keys = keys.transpose(1, 2)                 # [BxSxH] -> [BxHxS]

        score = torch.bmm(query, keys)                # [Bx1xH]x[BxHxS] -> [Bx1xS], batch_matrix_multiplication: bmm
        score = F.softmax(score.mul_(self.scale), dim=2)    # scale & normalize
        attention_value = torch.bmm(score, values).squeeze(1)    # [Bx1xS]x[BxSxH] -> [Bx1xH], 128개 각 값을 softmax 값을 기반으로 재조정

        return score.unsqueeze(1), attention_value


class SelfAttentionRNNRegressor(nn.Module):
    def __init__(self, embedding, encoder, attention, hidden_dim, n_actions):
        super(SelfAttentionRNNRegressor, self).__init__()
        self.embedding = embedding
        self.encoder = encoder
        self.attention = attention
        self.dense_decoder = nn.Linear(hidden_dim, n_actions)  # Dense
        size = 0
        for p in self.parameters():
            size += p.nelement()
        print('Total param size: {}'.format(size))

    def forward(self, input):
        outputs, hidden = self.encoder(self.embedding(input))
        # output: [32, 4, 128] or [1, 4, 128]
        # len(hidden): n_layers
        # hidden: [2, 32, 128] or [2, 1, 128] --> [n_layers, batch_size, hidden_size]

        hidden = hidden[-1]    # take the last layer's cell state
        # hidden: [32, 128] or [1, 128]

        # TODO: bidirectional은 False 라고 가정, 추후 True 고려하여 코딩 개선
        # if self.encoder.bidirectional:    # need to concat the last 2 hidden layers
        #     hidden = torch.cat([hidden[-1], hidden[-2]], dim=1)

        score, attention_value = self.attention(hidden, outputs, outputs)  # Q, K, V

        pred_value = self.dense_decoder(attention_value)  # [B, 1]

        return pred_value, score


class DDPGGruAttentionActor(nn.Module):
    def __init__(self, obs_size, hidden_size, n_actions, bidirectional, scale):
        super(DDPGGruAttentionActor, self).__init__()

        self.__name__ = "DDPGLstmAttentionActor"

        encoder = GruEncoder(
            embedding_dim=obs_size,
            hidden_dim=hidden_size,
            n_layers=2,
            dropout=0.0,
            bidirectional=bidirectional
        )

        embedding = NullEmbedding()

        attention_dim = hidden_size * 2 if bidirectional else hidden_size
        attention = Attention(attention_dim, attention_dim, attention_dim)  # Query, Key, Value

        self.net = SelfAttentionRNNRegressor(embedding, encoder, attention, attention_dim, n_actions=n_actions)

        self.net.apply(init_weights)

        self.scale = scale

    def forward(self, x):
        n, _ = self.net(x)
        t = torch.tanh(n)
        return t * self.scale


class DDPGGruAttentionCritic(nn.Module):
    def __init__(self, obs_size, hidden_size_1, hidden_size_2, n_actions, bidirectional):
        super(DDPGGruAttentionCritic, self).__init__()

        self.__name__ = "DDPGLstmAttentionCritic"

        encoder = GruEncoder(
            embedding_dim=obs_size,
            hidden_dim=hidden_size_1,
            n_layers=2,
            dropout=0.0,
            bidirectional=bidirectional
        )

        embedding = NullEmbedding()

        attention_dim = hidden_size_1 * 2 if bidirectional else hidden_size_1
        attention = Attention(attention_dim, attention_dim, attention_dim)

        self.obs_net = SelfAttentionRNNRegressor(embedding, encoder, attention, attention_dim, n_actions=1)

        self.out_net = nn.Sequential(
            nn.Linear(1 + n_actions, hidden_size_2),
            nn.ReLU(),
            nn.Linear(hidden_size_2, 1)
        )

        self.obs_net.apply(init_weights)
        self.out_net.apply(init_weights)

    def forward(self, x, a):
        obs, _ = self.obs_net(x)
        return self.out_net(torch.cat([obs, a], dim=1))


#####################################
## DDPGLstmAttention: End        ##
#####################################


#####################################
## DDPGLstm: Start        ##
#####################################

class DDPGGruActor(nn.Module):
    def __init__(self, obs_size, hidden_size_1, hidden_size_2, n_actions, bidirectional, scale):
        super(DDPGGruActor, self).__init__()

        self.__name__ = "DDPGLstmActor"

        self.net = GruEncoder(
            embedding_dim=obs_size,
            hidden_dim=hidden_size_1,
            n_layers=2,
            dropout=0.0,
            bidirectional=bidirectional
        )

        self.action_net = nn.Sequential(
            nn.Linear(hidden_size_1, hidden_size_2),
            nn.ReLU(),
            nn.Linear(hidden_size_2, n_actions),
        )

        self.net.apply(init_weights)
        self.action_net.apply(init_weights)

        self.scale = scale

    def forward(self, x):
        num_state_batch = x.shape[0]
        n_1, _ = self.net(x)
        n_2 = self.action_net(n_1[0:num_state_batch, -1, :])
        t = torch.tanh(n_2)
        return t * self.scale


class DDPGGruCritic(nn.Module):
    def __init__(self, obs_size, hidden_size_1, hidden_size_2, n_actions, bidirectional):
        super(DDPGGruCritic, self).__init__()

        self.__name__ = "DDPGLstmCritic"

        self.obs_net = GruEncoder(
            embedding_dim=obs_size,
            hidden_dim=hidden_size_1,
            n_layers=2,
            dropout=0.0,
            bidirectional=bidirectional
        )

        self.out_net = nn.Sequential(
            nn.Linear(hidden_size_1 + n_actions, hidden_size_2),
            nn.ReLU(),
            nn.Linear(hidden_size_2, 1)
        )

        self.obs_net.apply(init_weights)
        self.out_net.apply(init_weights)

    def forward(self, x, a):
        num_action_batch = a.shape[0]
        obs, _ = self.obs_net(x)
        obs = obs[0:num_action_batch, -1, :]
        return self.out_net(torch.cat([obs, a], dim=1)).squeeze(dim=0)

#####################################
## DDPGLstm: End                   ##
#####################################



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
            nn.Linear(hidden_size_1, hidden_size_2),
            nn.ReLU(),
        )

        self.out_net = nn.Sequential(
            nn.Linear(hidden_size_2 + n_actions, hidden_size_2),
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