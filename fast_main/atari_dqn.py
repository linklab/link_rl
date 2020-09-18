# https://github.com/AdrianHsu/breakout-Deep-Q-Network
#!/usr/bin/env python3
import time

import gym
import torch
import torch.optim as optim
import torch.multiprocessing as mp
import os
import warnings
import numpy as np

from fast_rl import dqn_model, rl_agent, actions, experience
from fast_rl.common import wrappers, utils, statistics
import atari_params
from fast_rl.dqn_model import insert_experience_into_buffer

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def make_env(params):
    env = gym.make(params.env_name)
    env = wrappers.wrap_dqn(env)
    return env


def play_func(env, params, net, device, exp_queue, args):
    action_selector = actions.EpsilonGreedyActionSelector(epsilon=params.epsilon_start)
    epsilon_tracker = actions.EpsilonTracker(
        action_selector=action_selector,
        eps_start=params.epsilon_start,
        eps_final=params.epsilon_final,
        eps_frames=params.epsilon_frames
    )
    agent = rl_agent.DQNAgent(net, action_selector, device=device)
    exp_source = experience.ExperienceSourceFirstLast(env, agent, gamma=params.gamma, steps_count=1)
    exp_source_iter = iter(exp_source)
    stat = statistics.Statistics(method="nature_dqn", args=args)

    action_count = []
    for _ in env.unwrapped.get_action_meanings():
        action_count.append(0)

    frame_idx = 0

    next_save_frame_idx = args.model_save_period

    with utils.AtariRewardTracker(stop_mean_episode_reward=params.stop_mean_episode_reward, stat=stat, args=args) as reward_tracker:
        while True:
            frame_idx += 1
            exp = next(exp_source_iter)
            action_count[exp.action] += 1
            exp_queue.put(exp)

            epsilon_tracker.udpate(frame_idx)

            episode_rewards = exp_source.pop_episode_reward_lst()

            if episode_rewards:
                solved, mean_episode_reward = reward_tracker.reward(episode_rewards[0], frame_idx, action_selector.epsilon, action_count)

                if frame_idx >= next_save_frame_idx:
                    dqn_model.save_model(args, net, frame_idx, mean_episode_reward)
                    next_save_frame_idx += args.model_save_period

                if solved:
                    break

    exp_queue.put(None)


def main():
    mp.set_start_method('spawn')

    args = utils.process_args()
    utils.print_args(args)

    device = torch.device("cuda" if args.cuda else "cpu")

    params = atari_params.HYPERPARAMS[args.env]
    # params.batch_size *= params.train_freq

    env = make_env(params)

    net = dqn_model.DQN(
        input_shape=env.observation_space.shape,
        n_actions=env.action_space.n
    ).to(device)
    print(net)
    print("ACTION MEANING: {0}".format(env.unwrapped.get_action_meanings()))

    tgt_net = rl_agent.TargetNet(net)

    buffer = experience.ExperienceReplayBuffer(experience_source=None, buffer_size=params.replay_size)
    #optimizer = optim.Adam(net.parameters(), lr=params.learning_rate)
    optimizer = optim.RMSprop(net.parameters(), lr=params.learning_rate, momentum=0.95, eps=0.01)

    exp_queue = mp.Queue(maxsize=params.train_freq * 2)
    play_proc = mp.Process(target=play_func, args=(env, params, net, device, exp_queue, args))
    play_proc.start()

    time.sleep(0.5)
    stat_for_model_loss = statistics.StatisticsForModelLoss(args=args)
    frame_idx = 0

    while play_proc.is_alive():
        frame_idx += params.train_freq
        for _ in range(params.train_freq):
            exp = exp_queue.get()
            if exp is None:
                play_proc.join()
                break
            insert_experience_into_buffer(exp, buffer)

        if len(buffer) < params.replay_initial:
            if args.draw_viz and frame_idx % 1000 == 0:
                stat_for_model_loss.draw_loss(frame_idx, 0.0)
            continue

        optimizer.zero_grad()
        batch = buffer.sample(params.batch_size)
        loss_v = dqn_model.calc_loss_dqn(batch, net, tgt_net, gamma=params.gamma, cuda=args.cuda, cuda_async=True)
        loss_v.backward()
        optimizer.step()

        if args.draw_viz and frame_idx % 1000 == 0:
            stat_for_model_loss.draw_loss(frame_idx, loss_v.item())

        if frame_idx % params.target_net_sync < params.train_freq:
            tgt_net.sync()


# python atari_dqn.py --env=pong --draw_viz=1 --cuda
# python atari_dqn.py --env=breakout --draw_viz=1 --cuda
if __name__ == "__main__":
    main()