import os

import torch
import torch.nn as nn
from torch.autograd import Variable

import numpy as np

from lib.fast_rl.common import MODEL_SAVE_DIR


def save_model(args, q_net, step, mean_episode_reward):
    model_save_filename = os.path.join(
        MODEL_SAVE_DIR, "{0}_{1}_{2}_{3}.pth".format(
            args.env, q_net.__name__, step, mean_episode_reward
        )
    )
    torch.save(q_net.state_dict(), model_save_filename)
    return model_save_filename


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
            torch.nn.init.kaiming_normal(m.weight)

    def _get_conv_out(self, shape):
        o = self.conv(Variable(torch.zeros(1, *shape)))
        return int(np.prod(o.size()))

    def forward(self, x):
        fx = x.float() / 256
        conv_out = self.conv(fx).view(fx.size()[0], -1)
        return self.fc(conv_out)


class DuelingDQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DuelingDQN, self).__init__()

        self.__name__ = "DuelingDQN"

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

        # self.conv.apply(self.init_weights)
        # self.fc_adv.apply(self.init_weights)
        # self.fc_val.apply(self.init_weights)

    def init_weights(self, m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            torch.nn.init.kaiming_normal(m.weight)

    def _get_conv_out(self, shape):
        o = self.conv(Variable(torch.zeros(1, *shape)))
        return int(np.prod(o.size()))

    def forward(self, x):
        fx = x.float() / 256
        conv_out = self.conv(fx).view(fx.size()[0], -1)
        val = self.fc_val(conv_out)
        adv = self.fc_adv(conv_out)
        return val + adv - adv.mean()


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


def unpack_batch(batch):
    states, actions, rewards, dones, last_states = [], [], [], [], []
    for exp in batch:
        state = np.array(exp.state, copy=False)
        states.append(state)
        actions.append(exp.action)
        rewards.append(exp.reward)
        dones.append(exp.last_state is None)
        if exp.last_state is None:
            last_states.append(state)       # the result will be masked anyway
        else:
            last_states.append(np.array(exp.last_state, copy=False))
    return np.array(states, copy=False), np.array(actions), np.array(rewards, dtype=np.float32), \
           np.array(dones, dtype=np.uint8), np.array(last_states, copy=False)


def insert_experience_into_buffer(exp, buffer):
    assert np.array_equal(exp.state.__array__()[1, :, :], exp.last_state.__array__()[0, :, :])
    assert np.array_equal(exp.state.__array__()[2, :, :], exp.last_state.__array__()[1, :, :])
    assert np.array_equal(exp.state.__array__()[3, :, :], exp.last_state.__array__()[2, :, :])

    extended_frames = np.zeros([5, 84, 84], dtype=np.uint8)
    extended_frames[0, :, :] = exp.state.__array__()[0, :, :]
    for i in range(1, 4):
        extended_frames[i, :, :] = exp.state.__array__()[i, :, :]

    if exp.last_state is not None:
        extended_frames[4, :, :] = exp.last_state.__array__()[3, :, :]

    buffer._add((extended_frames, exp.action, exp.reward, exp.last_state is None))


def unpack_batch_extended_frames(batch):
    states, actions, rewards, dones, last_states = [], [], [], [], []
    for transition in batch:
        extended_frames = transition[0]
        action = transition[1]
        reward = transition[2]
        done = transition[3]

        state = extended_frames[0:4, :, :]
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        dones.append(done)
        if not done:
            last_state = extended_frames[1:5, :, :]
        else:
            last_state = extended_frames[0:4, :, :]  # 어차피 추후에 무시됨
        last_states.append(last_state)

    return np.array(states, copy=False), np.array(actions), np.array(rewards, dtype=np.float32), \
           np.array(dones, dtype=np.uint8), np.array(last_states, copy=False)


def calc_loss_dqn(batch, net, tgt_net, gamma, cuda=False, cuda_async=False):
    states, actions, rewards, dones, next_states = unpack_batch_extended_frames(batch)
    #states, actions, rewards, dones, next_states = unpack_batch(batch)

    states_v = torch.tensor(states)
    next_states_v = torch.tensor(next_states)
    actions_v = torch.tensor(actions)
    rewards_v = torch.tensor(rewards)
    done_mask = torch.BoolTensor(dones)
    if cuda:
        states_v = states_v.cuda(non_blocking=cuda_async)
        next_states_v = next_states_v.cuda(non_blocking=cuda_async)
        actions_v = actions_v.cuda(non_blocking=cuda_async)
        rewards_v = rewards_v.cuda(non_blocking=cuda_async)
        done_mask = done_mask.cuda(non_blocking=cuda_async)

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

    expected_state_action_values = next_state_values.detach() * gamma + rewards_v

    return nn.MSELoss()(state_action_values, expected_state_action_values)


def calc_loss_double_dqn(batch, net, tgt_net, gamma, cuda=False, cuda_async=False):
    states, actions, rewards, dones, next_states = unpack_batch(batch)

    states_v = torch.tensor(states)
    next_states_v = torch.tensor(next_states)
    actions_v = torch.tensor(actions)
    rewards_v = torch.tensor(rewards)
    done_mask = torch.BoolTensor(dones)
    if cuda:
        states_v = states_v.cuda(non_blocking=cuda_async)
        next_states_v = next_states_v.cuda(non_blocking=cuda_async)
        actions_v = actions_v.cuda(non_blocking=cuda_async)
        rewards_v = rewards_v.cuda(non_blocking=cuda_async)
        done_mask = done_mask.cuda(non_blocking=cuda_async)

    actions_v = actions_v.unsqueeze(-1)
    state_action_vals = net(states_v).gather(1, actions_v)
    state_action_vals = state_action_vals.squeeze(-1)
    with torch.no_grad():
        next_state_acts = net(next_states_v).max(1)[1]
        next_state_acts = next_state_acts.unsqueeze(-1)
        next_state_vals = tgt_net.target_model(next_states_v).gather(1, next_state_acts).squeeze(-1)
        next_state_vals[done_mask] = 0.0
        exp_sa_vals = next_state_vals.detach() * gamma + rewards_v
    return nn.MSELoss()(state_action_vals, exp_sa_vals)


def calc_loss_per_double_dqn(batch, batch_weights, net, tgt_net, gamma, cuda=False, cuda_async=False):
    states, actions, rewards, dones, next_states = unpack_batch(batch)

    states_v = torch.tensor(states)
    next_states_v = torch.tensor(next_states)
    actions_v = torch.tensor(actions)
    rewards_v = torch.tensor(rewards)
    done_mask = torch.BoolTensor(dones)
    batch_weights_v = torch.tensor(batch_weights)
    if cuda:
        states_v = states_v.cuda(non_blocking=cuda_async)
        next_states_v = next_states_v.cuda(non_blocking=cuda_async)
        actions_v = actions_v.cuda(non_blocking=cuda_async)
        rewards_v = rewards_v.cuda(non_blocking=cuda_async)
        done_mask = done_mask.cuda(non_blocking=cuda_async)
        batch_weights_v = batch_weights_v.cuda(non_blocking=cuda_async)

    actions_v = actions_v.unsqueeze(-1)
    state_action_vals = net(states_v).gather(1, actions_v)
    state_action_vals = state_action_vals.squeeze(-1)
    with torch.no_grad():
        next_state_acts = net(next_states_v).max(1)[1]
        next_state_acts = next_state_acts.unsqueeze(-1)
        next_state_vals = tgt_net.target_model(next_states_v).gather(1, next_state_acts).squeeze(-1)
        next_state_vals[done_mask] = 0.0
        exp_sa_vals = next_state_vals.detach() * gamma + rewards_v
    losses_v = batch_weights_v * (state_action_vals - exp_sa_vals) ** 2
    return losses_v.mean(), (losses_v + 1e-5).data.cpu().numpy()
