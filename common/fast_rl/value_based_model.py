import glob
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
# from memory_profiler import profile


class NoisyLinear(nn.Linear):
    def __init__(self, in_features, out_features, sigma_init=0.017, bias=True):
        super(NoisyLinear, self).__init__(in_features, out_features, bias=bias)
        w = torch.full((out_features, in_features), sigma_init)
        self.sigma_weight = nn.Parameter(w)
        z = torch.zeros(out_features, in_features)
        self.register_buffer("epsilon_weight", z)
        if bias:
            w = torch.full((out_features,), sigma_init)
            self.sigma_bias = nn.Parameter(w)
            z = torch.zeros(out_features)
            self.register_buffer("epsilon_bias", z)
        self.reset_parameters()

    def reset_parameters(self):
        std = math.sqrt(3 / self.in_features)
        self.weight.data.uniform_(-std, std)
        self.bias.data.uniform_(-std, std)

    def forward(self, input):
        self.epsilon_weight.normal_()
        bias = self.bias
        if bias is not None:
            self.epsilon_bias.normal_()
            bias = bias + self.sigma_bias * self.epsilon_bias.data
        v = self.sigma_weight * self.epsilon_weight.data + self.weight
        return F.linear(input, v, bias)


# class NoisyLinear(nn.Module):
#     def __init__(self, in_features, out_features, std_init=0.4):
#         super(NoisyLinear, self).__init__()
#
#         self.in_features = in_features
#         self.out_features = out_features
#         self.std_init = std_init
#
#         self.weight_mu = nn.Parameter(torch.FloatTensor(out_features, in_features))
#         self.weight_sigma = nn.Parameter(torch.FloatTensor(out_features, in_features))
#         self.register_buffer('weight_epsilon', torch.FloatTensor(out_features, in_features))
#
#         self.bias_mu = nn.Parameter(torch.FloatTensor(out_features))
#         self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features))
#         self.register_buffer('bias_epsilon', torch.FloatTensor(out_features))
#
#         self.reset_parameters()
#         self.reset_noise()
#
#     def forward(self, x):
#         if self.training:
#             weight = self.weight_mu + self.weight_sigma.mul(Variable(self.weight_epsilon))
#             bias = self.bias_mu + self.bias_sigma.mul(Variable(self.bias_epsilon))
#         else:
#             weight = self.weight_mu
#             bias = self.bias_mu
#
#         return F.linear(x, weight, bias)
#
#     def reset_parameters(self):
#         mu_range = 1 / math.sqrt(self.weight_mu.size(1))
#
#         self.weight_mu.data.uniform_(-mu_range, mu_range)
#         self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.weight_sigma.size(1)))
#
#         self.bias_mu.data.uniform_(-mu_range, mu_range)
#         self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.bias_sigma.size(0)))
#
#     def reset_noise(self):
#         epsilon_in = self._scale_noise(self.in_features)
#         epsilon_out = self._scale_noise(self.out_features)
#
#         self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
#         self.bias_epsilon.copy_(self._scale_noise(self.out_features))
#
#     def _scale_noise(self, size):
#         x = torch.randn(size)
#         x = x.sign().mul(x.abs().sqrt())
#         return x



class NoisyDQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(NoisyDQN, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(input_shape)
        self.noisy_layers = [
            NoisyLinear(conv_out_size, 512),
            NoisyLinear(512, n_actions)
        ]
        self.fc = nn.Sequential(
            self.noisy_layers[0],
            nn.ReLU(),
            self.noisy_layers[1]
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        fx = x.float() / 256
        conv_out = self.conv(fx).view(fx.size()[0], -1)
        return self.fc(conv_out)

    def noisy_layers_sigma_snr(self):
        return [
            ((layer.weight ** 2).mean().sqrt() / (layer.sigma_weight ** 2).mean().sqrt()).item()
            for layer in self.noisy_layers
        ]


class DQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()

        self.__name__ = "DQN"

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(input_shape)
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

        self.conv.apply(self.init_weights)
        self.fc.apply(self.init_weights)

    def init_weights(self, m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            torch.nn.init.kaiming_normal_(m.weight)

    def _get_conv_out(self, shape):
        o = self.conv(Variable(torch.zeros(1, *shape)))
        return int(np.prod(o.size()))

    def forward(self, x):
        fx = x.float() / 255.
        conv_out = self.conv(fx).view(fx.size()[0], -1)
        return self.fc(conv_out)


class DuelingDQNCNN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DuelingDQNCNN, self).__init__()

        self.__name__ = "DuelingDQNCNN"

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(input_shape)
        self.fc_adv = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )
        self.fc_val = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

        self.conv.apply(self.init_weights)
        self.fc_adv.apply(self.init_weights)
        self.fc_val.apply(self.init_weights)

    def init_weights(self, m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            torch.nn.init.kaiming_normal_(m.weight)

    def _get_conv_out(self, shape):
        o = self.conv(Variable(torch.zeros(1, *shape)))
        return int(np.prod(o.size()))

    def forward(self, x):
        fx = x.float() / 255.
        conv_out = self.conv(fx).view(fx.size()[0], -1)
        val = self.fc_val(conv_out)
        adv = self.fc_adv(conv_out)
        return val + adv - adv.mean()


class RainbowDQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(RainbowDQN, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(input_shape)
        self.fc_adv = nn.Sequential(
            NoisyLinear(conv_out_size, 256),
            nn.ReLU(),
            NoisyLinear(256, n_actions)
        )
        self.fc_val = nn.Sequential(
            nn.Linear(conv_out_size, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        adv, val = self.adv_val(x)
        return val + (adv - adv.mean(dim=1, keepdim=True))

    def adv_val(self, x):
        fx = x.float() / 256
        conv_out = self.conv(fx).view(fx.size()[0], -1)
        return self.fc_adv(conv_out), self.fc_val(conv_out)


class DQNMLP(nn.Module):
    def __init__(self, obs_size, hidden_size, n_actions):
        super(DQNMLP, self).__init__()

        self.__name__ = "DQNMLP"

        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions)
        )

    def forward(self, x):
        # CartPole is stupid -- they return double observations, rather than standard floats, so, the cast here
        return self.net(x.float())


class DuelingDQNMLP(nn.Module):
    def __init__(self, obs_size, hidden_size_1, hidden_size_2, n_actions):
        super(DuelingDQNMLP, self).__init__()

        self.__name__ = "DuelingDQNMLP"

        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size_1),
            nn.ReLU(),
            nn.Linear(hidden_size_1, hidden_size_2),
            nn.ReLU()
        )

        self.fc_adv = nn.Sequential(
            nn.Linear(hidden_size_2, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )
        self.fc_val = nn.Sequential(
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
        val = self.fc_val(net_out)
        adv = self.fc_adv(net_out)
        return val + adv - adv.mean()


class RainbowDQNMLP(nn.Module):
    def __init__(self, obs_size, hidden_size_1, hidden_size_2, n_actions):
        super(RainbowDQNMLP, self).__init__()

        self.__name__ = "RainbowDQNMLP"

        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size_1),
            nn.ReLU(),
            nn.Linear(hidden_size_1, hidden_size_2),
            nn.ReLU()
        )

        self.noisy_linear_1 = NoisyLinear(hidden_size_2, 128)
        self.noisy_linear_2 = NoisyLinear(128, n_actions)

        self.fc_adv = nn.Sequential(
            self.noisy_linear_1,
            nn.ReLU(),
            self.noisy_linear_2
        )
        self.fc_val = nn.Sequential(
            nn.Linear(hidden_size_2, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    #     self.net.apply(self.init_weights)
    #     self.fc_adv.apply(self.init_weights)
    #     self.fc_val.apply(self.init_weights)
    #
    # def init_weights(self, m):
    #     if type(m) == nn.Linear or type(m) == nn.Conv2d:
    #         torch.nn.init.kaiming_normal_(m.weight)

    def forward(self, x):
        if torch.is_tensor(x):
            x = x.to(torch.float32)
        else:
            x = torch.tensor(x, dtype=torch.float32)
        net_out = self.net(x)
        val = self.fc_val(net_out)
        adv = self.fc_adv(net_out)
        return val + adv - adv.mean()


def unpack_batch(batch):
    states, actions, rewards, dones, last_states, last_steps = [], [], [], [], [], []

    for exp in batch:
        state = np.array(exp.state, copy=False)
        states.append(state)
        actions.append(exp.action)
        rewards.append(exp.reward)
        dones.append(exp.last_state is None)
        last_steps.append(exp.last_step)
        if exp.last_state is None:
            last_states.append(state)  # the result will be masked anyway
        else:
            last_states.append(np.array(exp.last_state, copy=False))
    return np.array(states, copy=False), np.array(actions), np.array(rewards, dtype=np.float32), \
           np.array(dones, dtype=np.uint8), np.array(last_states, copy=False), np.array(last_steps)


def insert_experience_into_buffer(exp, buffer):
    # assert np.array_equal(exp.state.__array__()[1, :, :], exp.last_state.__array__()[0, :, :])
    # assert np.array_equal(exp.state.__array__()[2, :, :], exp.last_state.__array__()[1, :, :])
    # assert np.array_equal(exp.state.__array__()[3, :, :], exp.last_state.__array__()[2, :, :])

    extended_frames = np.zeros([5, 84, 84], dtype=np.uint8)
    extended_frames[0, :, :] = exp.state.__array__()[0, :, :]
    for i in range(1, 4):
        extended_frames[i, :, :] = exp.state.__array__()[i, :, :]

    if exp.last_state is not None:
        extended_frames[4, :, :] = exp.last_state.__array__()[3, :, :]

    buffer._add((extended_frames, exp.action, exp.reward, exp.last_state is None, exp.last_step))


def unpack_batch_extended_frames(batch):
    states, actions, rewards, dones, last_states, last_steps = [], [], [], [], [], []
    for transition in batch:
        extended_frames = transition[0]
        action = transition[1]
        reward = transition[2]
        done = transition[3]
        last_step = transition[4]

        state = extended_frames[0:4, :, :]
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        dones.append(done)
        last_steps.append(last_step)
        if not done:
            last_state = extended_frames[1:5, :, :]
        else:
            last_state = extended_frames[0:4, :, :]  # 어차피 추후에 무시됨
        last_states.append(last_state)

    return np.array(states, copy=False), np.array(actions), np.array(rewards, dtype=np.float32), \
           np.array(dones, dtype=np.uint8), np.array(last_states, copy=False), np.array(last_steps)


def unpack_batch_for_n_step(buffer, batch, batch_indices, params):
    states, actions, rewards, dones, next_states, last_steps = [], [], [], [], [], []
    for idx, exp in enumerate(batch):
        state = np.array(exp.state, copy=False)
        states.append(state)
        actions.append(exp.action)

        n_step_rewards = 0
        gamma = 1
        current_exp = exp
        for i in range(params.N_STEP):
            n_step_rewards += gamma * current_exp.reward
            next_exp = buffer[(batch_indices[idx] + i + 1) % params.REPLAY_BUFFER_SIZE]

            if current_exp.done:
                rewards.append(n_step_rewards)
                next_states.append(np.array(next_exp.state, copy=False))
                dones.append(True)
                last_steps.append(i + 1)
                break
            else:
                if i == params.N_STEP - 1:
                    rewards.append(n_step_rewards)
                    next_states.append(np.array(next_exp.state, copy=False))
                    dones.append(False)
                    last_steps.append(i + 1)

            current_exp = next_exp
            gamma *= params.GAMMA

    return np.array(states, copy=False), np.array(actions), np.array(rewards, dtype=np.float32), \
           np.array(dones, dtype=np.uint8), np.array(next_states, copy=False), np.array(last_steps)


def unpack_batch_for_omega(buffer, batch, batch_indices, params):
    states, actions, rewards, done_mask, next_states = [], [], [], [], []
    for idx, exp in enumerate(batch):
        state = np.array(exp.state, copy=False)
        states.append(state)
        actions.append(exp.action)

        n_step_rewards = []
        current_exp = exp
        for i in range(params.OMEGA_WINDOW_SIZE):
            n_step_rewards.append(current_exp.reward)
            next_exp = buffer[(batch_indices[idx] + i + 1) % params.REPLAY_BUFFER_SIZE]
            next_states.append(np.array(next_exp.state, copy=False))

            if current_exp.done:
                done_mask.append(0)
                break
            else:
                if i == params.OMEGA_WINDOW_SIZE - 1:
                    done_mask.append(1)

            current_exp = next_exp

        rewards.append(n_step_rewards)

    return np.array(states, copy=False), np.array(actions), np.array(rewards), np.array(done_mask), np.array(next_states, copy=False)


def calc_loss_dqn(batch, net, tgt_net, gamma, cuda=False, cuda_async=False):
    # states, actions, rewards, dones, next_states, last_steps = unpack_batch_extended_frames(batch)
    states, actions, rewards, dones, next_states, last_steps = unpack_batch(batch)

    states_v = torch.tensor(states)
    next_states_v = torch.tensor(next_states)
    actions_v = torch.tensor(actions)
    rewards_v = torch.tensor(rewards)
    done_mask = torch.BoolTensor(dones)
    last_steps_v = torch.tensor(last_steps)
    if cuda:
        states_v = states_v.cuda(non_blocking=cuda_async)
        next_states_v = next_states_v.cuda(non_blocking=cuda_async)
        actions_v = actions_v.cuda(non_blocking=cuda_async)
        rewards_v = rewards_v.cuda(non_blocking=cuda_async)
        done_mask = done_mask.cuda(non_blocking=cuda_async)
        last_steps_v = last_steps_v.cuda(non_blocking=cuda_async)

    # https://subscription.packtpub.com/book/data/9781838826994/6/ch06lvl1sec45/dqn-on-pong
    # We pass observations to the first model and extract the specific Q - values for the taken actions using the gather() tensor operation.
    # The first argument to the gather() call is a dimension index that we want to perform gathering on.
    # In our case, it is equal to 1, which corresponds to actions.
    # The second argument is a tensor of indices of elements to be chosen.
    # Extra unsqueeze() and squeeze() calls are required to compute the index argument for the gather functions,
    # and to get rid of the extra dimensions that we created, respectively.
    # The index should have the same number of dimensions as the data we are processing.
    # In Figure 6.3, you can see an illustration of what gather() does on the example case, with a batch of six entries and four actions:
    state_action_values = net(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)

    with torch.no_grad():
        next_state_values = tgt_net.target_model(next_states_v).max(1)[0]
        next_state_values[done_mask] = 0.0

    expected_state_action_values = next_state_values.detach() * (gamma ** last_steps_v) + rewards_v

    # return nn.MSELoss()(state_action_values, expected_state_action_values)
    return F.smooth_l1_loss(state_action_values, expected_state_action_values)


def calc_loss_double_dqn(batch, net, tgt_net, gamma, cuda=False, cuda_async=False):
    states, actions, rewards, dones, next_states, last_steps = unpack_batch(batch)

    states_v = torch.tensor(states)
    next_states_v = torch.tensor(next_states)
    actions_v = torch.tensor(actions)
    rewards_v = torch.tensor(rewards)
    done_mask = torch.BoolTensor(dones)
    last_steps_v = torch.tensor(last_steps)
    if cuda:
        states_v = states_v.cuda(non_blocking=cuda_async)
        next_states_v = next_states_v.cuda(non_blocking=cuda_async)
        actions_v = actions_v.cuda(non_blocking=cuda_async)
        rewards_v = rewards_v.cuda(non_blocking=cuda_async)
        done_mask = done_mask.cuda(non_blocking=cuda_async)
        last_steps_v = last_steps_v.cuda(non_blocking=cuda_async)

    actions_v = actions_v.unsqueeze(-1)
    state_action_values = net(states_v).gather(1, actions_v)
    state_action_values = state_action_values.squeeze(-1)
    with torch.no_grad():
        next_state_acts = net(next_states_v).max(1)[1]
        next_state_acts = next_state_acts.unsqueeze(-1)
        next_state_vals = tgt_net.target_model(next_states_v).gather(1, next_state_acts).squeeze(-1)
        next_state_vals[done_mask] = 0.0

    exp_sa_vals = next_state_vals.detach() * (gamma ** last_steps_v) + rewards_v

    # return nn.MSELoss()(state_action_values, exp_sa_vals)
    return F.smooth_l1_loss(state_action_values, exp_sa_vals)


# -m memory_profiler
# @profile
def calc_loss_per_double_dqn(buffer, batch, batch_indices, batch_weights, net, tgt_net, params, cuda=False,
                             cuda_async=False):
    if params.NEXT_STATE_IN_TRAJECTORY:
        states, actions, rewards, dones, next_states, last_steps = unpack_batch(batch)
    else:
        states, actions, rewards, dones, next_states, last_steps = unpack_batch_for_n_step(
            buffer, batch, batch_indices, params
        )

    states_v = torch.tensor(states)
    next_states_v = torch.tensor(next_states)
    actions_v = torch.tensor(actions)
    rewards_v = torch.tensor(rewards)
    done_mask = torch.BoolTensor(dones)
    last_steps_v = torch.tensor(last_steps)
    batch_weights_v = torch.tensor(batch_weights)
    if cuda:
        states_v = states_v.cuda(non_blocking=cuda_async)
        next_states_v = next_states_v.cuda(non_blocking=cuda_async)
        actions_v = actions_v.cuda(non_blocking=cuda_async)
        rewards_v = rewards_v.cuda(non_blocking=cuda_async)
        done_mask = done_mask.cuda(non_blocking=cuda_async)
        last_steps_v = last_steps_v.cuda(non_blocking=cuda_async)
        batch_weights_v = batch_weights_v.cuda(non_blocking=cuda_async)

    actions_v = actions_v.unsqueeze(-1)
    state_action_values = net(states_v).gather(1, actions_v)
    state_action_values = state_action_values.squeeze(-1)

    with torch.no_grad():
        next_state_actions = net(next_states_v).max(1)[1]
        next_state_actions = next_state_actions.unsqueeze(-1)
        next_state_values = tgt_net.target_model(next_states_v).gather(1, next_state_actions).squeeze(-1)
        next_state_values[done_mask] = 0.0

    expected_state_action_values = next_state_values.detach() * (params.GAMMA ** last_steps_v) + rewards_v
    losses_v = batch_weights_v * F.smooth_l1_loss(state_action_values, expected_state_action_values)
    return losses_v.mean(), (losses_v + 1e-5)


def calc_loss_per_double_dqn_for_omega(buffer, batch, batch_indices, batch_weights, net, tgt_net, params, cuda=False,
                                       cuda_async=False):
    states, actions, rewards, done_mask, next_states = unpack_batch_for_omega(buffer, batch, batch_indices, params)

    states_v = torch.tensor(states)
    next_states_v = torch.tensor(next_states)
    actions_v = torch.tensor(actions)
    batch_weights_v = torch.tensor(batch_weights)
    if cuda:
        states_v = states_v.cuda(non_blocking=cuda_async)
        next_states_v = next_states_v.cuda(non_blocking=cuda_async)
        actions_v = actions_v.cuda(non_blocking=cuda_async)
        batch_weights_v = batch_weights_v.cuda(non_blocking=cuda_async)

    actions_v = actions_v.unsqueeze(-1)
    state_action_values = net(states_v).gather(1, actions_v)
    state_action_values = state_action_values.squeeze(-1)

    with torch.no_grad():
        # for double DQN
        next_state_actions = net(next_states_v).max(1)[1]
        next_state_actions = next_state_actions.unsqueeze(-1)
        next_state_values = tgt_net.target_model(next_states_v).gather(1, next_state_actions).squeeze(-1)

    expected_state_action_values = calc_omega_return(rewards, done_mask, next_state_values.detach().cpu().numpy(), params)
    expected_state_action_values = torch.tensor(expected_state_action_values, dtype=torch.float32)
    if cuda:
        expected_state_action_values = expected_state_action_values.cuda(non_blocking=cuda_async)

    losses_v = batch_weights_v * F.smooth_l1_loss(state_action_values, expected_state_action_values)
    return losses_v.mean(), (losses_v + 1e-5)


def calc_omega_return(rewards, done_mask, next_state_values, params):
    idx_count = 0
    target_q_values = []
    for batch_idx in range(params.BATCH_SIZE):
        n_step_target_list = []
        n_step_reward_sum_list = []
        reward_sum = 0
        gamma = 1
        for idx, reward in enumerate(rewards[batch_idx]):
            reward_sum += gamma * reward
            n_step_reward_sum_list.append(reward_sum)
            gamma *= params.GAMMA
        gamma = params.GAMMA
        for i in range(len(rewards[batch_idx])):
            n_step_target_list.append(n_step_reward_sum_list[i] + gamma * next_state_values[idx_count] *
                                      (done_mask[batch_idx] if i == len(rewards[batch_idx]) - 1 else 1))
            gamma *= params.GAMMA
            idx_count += 1

        avg = sum(n_step_target_list) / len(n_step_target_list)
        max_n_step_target = max(n_step_target_list)
        beta = (max_n_step_target - avg) / (max_n_step_target - min(n_step_target_list) + 0.00001)
        target_q_values.append((1 - beta) * avg + beta * max_n_step_target)

    # target_q_values = torch.tensor(target_q_values, dtype=torch.float32)
    return target_q_values
