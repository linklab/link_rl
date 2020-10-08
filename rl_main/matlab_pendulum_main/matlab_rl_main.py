#!/usr/bin/env python3
import time

import gym
import torch
import torch.multiprocessing as mp
from torch import optim
import os

from common.environments.matlab.matlabenv import MatlabRotaryInvertedPendulumEnv
print(torch.__version__)

from common.fast_rl import actions, experience, value_based_model, rl_agent
from common.fast_rl.common import statistics, utils
from config.parameters import PARAMETERS as params


cuda = False
# env_name = 'CartPole-v1'


MODEL_SAVE_DIR = os.path.join(".", "saved_models")
if not os.path.exists(MODEL_SAVE_DIR):
    os.makedirs(MODEL_SAVE_DIR)


os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

if torch.cuda.is_available():
    device = torch.device("cuda" if params.CUDA else "cpu")
else:
    device = torch.device("cpu")


def play_func(exp_queue, env, net):
    env.start()
    action_selector = actions.EpsilonGreedyActionSelector(epsilon=params.EPSILON_INIT)

    epsilon_tracker = actions.EpsilonTracker(
        action_selector=action_selector,
        eps_start=params.EPSILON_INIT,
        eps_final=params.EPSILON_MIN,
        eps_frames=params.EPSILON_MIN_STEP
    )

    agent = rl_agent.DQNAgent(net, action_selector, device=device)
    experience_source = experience.ExperienceSourceFirstLast(
        env, agent, params.GAMMA, steps_count=params.N_STEP
    )
    exp_source_iter = iter(experience_source)

    stat = statistics.StatisticsForValueBasedRL(method="nature_dqn")

    frame_idx = 0
    next_save_frame_idx = params.MODEL_SAVE_STEP_PERIOD

    with utils.RewardTracker(params.STOP_MEAN_EPISODE_REWARD, params.AVG_EPISODE_SIZE_FOR_STAT, True, params.DRAW_VIZ, stat) as reward_tracker:
        while True:
            frame_idx += 1
            exp = next(exp_source_iter)
            exp_queue.put(exp)

            epsilon_tracker.udpate(frame_idx)

            episode_rewards = experience_source.pop_episode_reward_lst()
            if episode_rewards:
                solved, mean_episode_reward = reward_tracker.set_episode_reward(
                    episode_rewards[0], frame_idx, action_selector.epsilon
                )

                if frame_idx >= next_save_frame_idx:
                    rl_agent.save_model(MODEL_SAVE_DIR, params.ENV_NAME, net.__name__, net, frame_idx, mean_episode_reward)
                    next_save_frame_idx += params.MODEL_SAVE_STEP_PERIOD

                if solved:
                    rl_agent.save_model(MODEL_SAVE_DIR, params.ENV_NAME, net.__name__, net, frame_idx, mean_episode_reward)
                    break

    exp_queue.put(None)


def main():
    mp.set_start_method('spawn')

    # env = gym.make(env_name)
    env = MatlabRotaryInvertedPendulumEnv()

    net = value_based_model.DuelingDQNMLP(
        obs_size=4,
        hidden_size_1=128, hidden_size_2=128,
        n_actions=7
    ).to(device)

    print(net)
    print(env)
    tgt_net = rl_agent.TargetNet(net)

    buffer = experience.PrioReplayBuffer(exp_source=None, buf_size=params.REPLAY_BUFFER_SIZE)
    optimizer = optim.Adam(net.parameters(), lr=params.LEARNING_RATE)

    exp_queue = mp.Queue(maxsize=params.TRAIN_STEP_FREQ * 2)
    play_proc = mp.Process(target=play_func, args=(exp_queue, env, net))
    play_proc.start()

    time.sleep(0.5)
    stat_for_model_loss = statistics.StatisticsForValueBasedOptimization()
    frame_idx = 0

    while play_proc.is_alive():
        frame_idx += params.TRAIN_STEP_FREQ
        for _ in range(params.TRAIN_STEP_FREQ):
            exp = exp_queue.get()
            if exp is None:
                play_proc.join()
                break
            buffer._add(exp)

        if len(buffer) < params.MIN_REPLAY_SIZE_FOR_TRAIN:
            if params.DRAW_VIZ and frame_idx % 100 == 0:
                stat_for_model_loss.draw_optimization_performance(frame_idx, 0.0)
            continue

        optimizer.zero_grad()
        batch, batch_indices, batch_weights = buffer.sample(params.BATCH_SIZE)
        loss_v, sample_prios = value_based_model.calc_loss_per_double_dqn(
            buffer.buffer, batch, batch_indices, batch_weights, net, tgt_net, params, cuda=cuda, cuda_async=True
        )
        loss_v.backward()
        optimizer.step()
        buffer.update_priorities(batch_indices, sample_prios)
        buffer.update_beta(frame_idx)

        if params.DRAW_VIZ and frame_idx % 100 == 0:
            stat_for_model_loss.draw_optimization_performance(frame_idx, loss_v.item())

        if frame_idx % params.TARGET_NET_SYNC_STEP_PERIOD < params.TRAIN_STEP_FREQ:
            tgt_net.sync()


if __name__ == "__main__":
    main()