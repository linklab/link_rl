#!/usr/bin/env python3
import gym
import numpy as np
import torch
import torch.optim as optim
import os
import warnings

from rl_main.fast_main import atari_params
from common.fast_rl import experience, rl_agent, dqn_model, actions
from common.fast_rl.common import utils
from common.fast_rl.common import statistics, wrappers

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


if __name__ == "__main__":
    args = utils.process_args()
    utils.print_args(args)

    device = torch.device("cuda" if args.cuda else "cpu")

    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    params = atari_params.HYPERPARAMS[args.env]
    params.batch_size *= params.train_freq

    env = gym.make(params.env_name)
    env = wrappers.wrap_dqn(env)
    if args.seed is not None:
        env.seed(args.seed)

    suffix = "" if args.seed is None else "_seed=%s" % args.seed
    net = dqn_model.DQN(
        input_shape=env.observation_space.shape,
        n_actions=env.action_space.n
    ).to(device)
    print(net)

    tgt_net = rl_agent.TargetNet(net)

    action_selector = actions.EpsilonGreedyActionSelector(epsilon=params.epsilon_start)
    epsilon_tracker = actions.EpsilonTracker(
        action_selector=action_selector,
        eps_start=params.epsilon_start,
        eps_final=params.epsilon_final,
        eps_frames=params.epsilon_frames
    )
    agent = rl_agent.DQNAgent(net, action_selector, device=device)

    exp_source = experience.ExperienceSourceFirstLast(env, agent, gamma=params.gamma, steps_count=1)
    buffer = experience.ExperienceReplayBuffer(exp_source, buffer_size=params.replay_size)
    optimizer = optim.Adam(net.parameters(), lr=params.learning_rate)

    stat = statistics.Statistics(method="nature_dqn")
    stat_for_model_loss = statistics.StatisticsForModelLoss()

    action_count = []
    for _ in env.unwrapped.get_action_meanings():
        action_count.append(0)

    frame_idx = 0

    next_save_frame_idx = args.model_save_period

    with utils.AtariRewardTracker(
            stop_mean_episode_reward=params.stop_mean_episode_reward,
            average_size_for_stats=params.average_size_for_stats,
            draw_viz=params.draw_viz, stat=stat) as reward_tracker:
        while True:
            frame_idx += params.train_freq
            buffer.populate_stacked_experience(params.train_freq)
            epsilon_tracker.udpate(frame_idx)

            episode_rewards = exp_source.pop_episode_reward_lst()
            if episode_rewards:
                solved, mean_episode_reward = reward_tracker.reward(
                    episode_rewards[0], frame_idx, action_selector.epsilon, action_count
                )

                if frame_idx >= next_save_frame_idx:
                    dqn_model.save_model(".", args.env_name, net.__name__, net, frame_idx, mean_episode_reward)
                    next_save_frame_idx += args.model_save_period

                if solved:
                    break

            if len(buffer) < params.replay_initial:
                continue

            optimizer.zero_grad()
            batch = buffer.sample(params.batch_size)
            loss_v = dqn_model.calc_loss_dqn(batch, net, tgt_net, gamma=params.gamma, cuda=args.cuda)
            loss_v.backward()
            optimizer.step()

            if args.draw_viz and frame_idx % 1000 == 0:
                stat_for_model_loss.draw_loss(frame_idx, loss_v.item())

            if frame_idx % params.target_net_sync < params.train_freq:
                tgt_net.sync()