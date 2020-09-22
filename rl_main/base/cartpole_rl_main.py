#!/usr/bin/env python3
import time

import gym
import torch
import torch.multiprocessing as mp
from torch import optim
import os

print(torch.__version__)

from common.environments.gym.cartpole import CartPole_v0
from common.fast_rl import actions, experience, dqn_model, rl_agent
from common.fast_rl.common import statistics, utils

cuda = False
epsilon_start = 1.0
epsilon_final = 0.01
epsilon_frames = 1000
gamma = 0.99
n_step = 1
stop_mean_episode_reward = 195
average_size_for_stats = 10
draw_viz = 1

train_freq = 2
batch_size = 32
batch_size *= train_freq
replay_size = 50000
learning_rate = 0.001
replay_initial = 100
target_net_sync = 50

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
device = torch.device("cuda" if cuda else "cpu")


def play_func(exp_queue, env, net):
    action_selector = actions.EpsilonGreedyActionSelector(epsilon=epsilon_start)

    epsilon_tracker = actions.EpsilonTracker(
        action_selector=action_selector,
        eps_start=epsilon_start,
        eps_final=epsilon_final,
        eps_frames=epsilon_frames
    )

    agent = rl_agent.DQNAgent(net, action_selector, device=device)

    experience_source = experience.ExperienceSourceFirstLast(
        env, agent, gamma=gamma, steps_count=n_step
    )

    exp_source_iter = iter(experience_source)

    stat = statistics.Statistics(method="nature_dqn")

    frame_idx = 0

    with utils.AtariRewardTracker(stop_mean_episode_reward, average_size_for_stats, draw_viz, stat) as reward_tracker:
        while True:
            frame_idx += 1
            exp = next(exp_source_iter)
            exp_queue.put(exp)

            epsilon_tracker.udpate(frame_idx)

            episode_rewards = experience_source.pop_episode_reward_lst()
            if episode_rewards:
                solved, mean_episode_reward = reward_tracker.reward(
                    episode_rewards[0], frame_idx, action_selector.epsilon
                )

                if solved:
                    break

    exp_queue.put(None)


def main():
    mp.set_start_method('spawn')

    #env = CartPole_v0()
    env = gym.make("CartPole-v0")

    net = dqn_model.DuelingDQNMLP(
        obs_size=4,
        hidden_size_1=128, hidden_size_2=128,
        n_actions=2
    ).to(device)
    print(net)
    tgt_net = rl_agent.TargetNet(net)

    buffer = experience.PrioReplayBuffer(exp_source=None, buf_size=replay_size)
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    exp_queue = mp.Queue(maxsize=train_freq * 2)
    play_proc = mp.Process(target=play_func, args=(exp_queue, env, net))
    play_proc.start()

    time.sleep(0.5)
    stat_for_model_loss = statistics.StatisticsForModelLoss()
    frame_idx = 0

    while play_proc.is_alive():
        frame_idx += train_freq
        for _ in range(train_freq):
            exp = exp_queue.get()
            if exp is None:
                play_proc.join()
                break
            buffer._add(exp)

        if len(buffer) < replay_initial:
            if draw_viz and frame_idx % 100 == 0:
                stat_for_model_loss.draw_loss(frame_idx, 0.0)
            continue

        optimizer.zero_grad()
        batch, batch_indices, batch_weights = buffer.sample(batch_size)
        loss_v, sample_prios = dqn_model.calc_loss_per_double_dqn(
            batch, batch_weights, net, tgt_net, gamma=gamma, cuda=cuda, cuda_async=True
        )
        loss_v.backward()
        optimizer.step()
        buffer.update_priorities(batch_indices, sample_prios)
        buffer.update_beta(frame_idx)

        if draw_viz and frame_idx % 100 == 0:
            stat_for_model_loss.draw_loss(frame_idx, loss_v.item())

        if frame_idx % target_net_sync < train_freq:
            tgt_net.sync()


if __name__ == "__main__":
    main()