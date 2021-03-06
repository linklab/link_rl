#!/usr/bin/env python3
import gym
import ptan
import argparse
import random
import numpy as np

import torch
import torch.optim as optim
import torch.nn.functional as F

from ignite.engine import Engine

from lib import common, dqn_extra
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

NAME = "07_distrib"

Vmin, Vmax, N_ATOMS = -10, 10, 51


def calc_loss(batch, net, target_network, gamma, device="cpu"):
    states, actions, rewards, dones, next_states = common.unpack_batch(batch)
    batch_size = len(batch)

    states_v = torch.tensor(states).to(device)
    actions_v = torch.tensor(actions).to(device)
    next_states_v = torch.tensor(next_states).to(device)

    # next state distribution
    next_distribution_v, next_q_values_v = target_network.both(next_states_v)
    next_actions = next_q_values_v.max(1)[1].data.cpu().numpy()
    next_distribution = target_network.apply_softmax(next_distribution_v)
    next_distribution = next_distribution.data.cpu().numpy()

    next_best_distribution = next_distribution[range(batch_size), next_actions]        # next_distribution: (32, 6, 51)
    dones = dones.astype(np.bool)

    projected_distribution = dqn_extra.distribution_projection(
        next_best_distribution, rewards, dones, Vmin, Vmax, N_ATOMS, gamma
    )

    distribution_v = net(states_v)
    selected_action_distribution = distribution_v[range(batch_size), actions_v.data]
    state_log_softmax_v = F.log_softmax(selected_action_distribution, dim=1)
    projected_distribution_v = torch.tensor(projected_distribution).to(device)

    loss_v = -state_log_softmax_v * projected_distribution_v
    return loss_v.sum(dim=1).mean()


if __name__ == "__main__":
    random.seed(common.SEED)
    torch.manual_seed(common.SEED)
    params = common.HYPERPARAMS['pong']
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action="store_true", help="Enable cuda")
    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")

    env = gym.make(params.env_name)
    env = ptan.common.wrappers.wrap_dqn(env)
    env.seed(common.SEED)

    net = dqn_extra.DistributionalDQN(env.observation_space.shape, env.action_space.n).to(device)
    print(net)
    target_network = ptan.agent.TargetNet(net)
    selector = ptan.actions.EpsilonGreedyDQNActionSelector(epsilon=params.epsilon_start)
    epsilon_tracker = common.EpsilonTracker(selector, params)
    agent = ptan.agent.DQNAgent(lambda x: net.q_values(x), selector, device=device)

    experience_source = ptan.experience.ExperienceSourceFirstLast(env, agent, gamma=params.gamma)
    buffer = ptan.experience.ExperienceReplayBuffer(experience_source, buffer_size=params.replay_size)
    optimizer = optim.Adam(net.parameters(), lr=params.learning_rate)

    def process_batch(engine, batch):
        optimizer.zero_grad()
        loss_v = calc_loss(batch, net, target_network.target_model, gamma=params.gamma, device=device)
        loss_v.backward()
        optimizer.step()
        epsilon_tracker.frame(engine.state.iteration)
        if engine.state.iteration % params.target_net_sync == 0:
            target_network.sync()

        return {
            "loss": loss_v.item(),
            "epsilon": selector.epsilon,
        }

    engine = Engine(process_batch)
    common.setup_ignite(engine, params, experience_source, NAME)
    engine.run(common.batch_generator(buffer, params.replay_initial, params.batch_size))
