#!/usr/bin/env python3
import gym
import numpy as np
import torch
import torch.optim as optim
import os
import warnings

from common import common_utils
from common.common_utils import make_atari_env
from common.fast_rl import experience, rl_agent, dqn_model, actions
from common.fast_rl.common import utils
from common.fast_rl.common import statistics, wrappers

##### NOTE #####
from config.parameters import PARAMETERS as params
##### NOTE #####

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

device = torch.device("cuda" if params.CUDA else "cpu")

MODEL_SAVE_DIR = os.path.join(".", "saved_models")
if not os.path.exists(MODEL_SAVE_DIR):
    os.makedirs(MODEL_SAVE_DIR)


if __name__ == "__main__":
    common_utils.print_fast_rl_params(params)

    params.BATCH_SIZE *= params.TRAIN_STEP_FREQ

    env = make_atari_env(params.ENVIRONMENT_ID.value, seed=params.SEED)

    if params.SEED is not None:
        env.seed(params.SEED)

    suffix = "" if params.SEED is None else "_seed=%s" % params.SEED

    net = dqn_model.DQN(
        input_shape=env.observation_space.shape,
        n_actions=env.action_space.n
    ).to(device)
    print(net)

    tgt_net = rl_agent.TargetNet(net)

    action_selector = actions.EpsilonGreedyActionSelector(epsilon=params.EPSILON_INIT)
    epsilon_tracker = actions.EpsilonTracker(
        action_selector=action_selector,
        eps_start=params.EPSILON_INIT,
        eps_final=params.EPSILON_MIN,
        eps_frames=params.EPSILON_MIN_STEP
    )
    agent = rl_agent.DQNAgent(net, action_selector, device=device)

    exp_source = experience.ExperienceSourceFirstLast(env, agent, gamma=params.GAMMA, steps_count=1)
    buffer = experience.ExperienceReplayBuffer(exp_source, buffer_size=params.REPLAY_BUFFER_SIZE)
    optimizer = optim.Adam(net.parameters(), lr=params.LEARNING_RATE)

    if params.DRAW_VIZ:
        stat = statistics.StatisticsForValueBasedRL(method="nature_dqn")
        stat_for_model_loss = statistics.StatisticsForValueBasedOptimization()
    else:
        stat = None
        stat_for_model_loss = None

    action_count = []
    for _ in env.unwrapped.get_action_meanings():
        action_count.append(0)

    frame_idx = 0

    next_save_frame_idx = params.MODEL_SAVE_STEP_PERIOD

    with utils.RewardTracker(
            stop_mean_episode_reward=params.STOP_MEAN_EPISODE_REWARD,
            average_size_for_stats=params.AVG_EPISODE_SIZE_FOR_STAT,
            draw_viz=params.DRAW_VIZ, stat=stat) as reward_tracker:
        while frame_idx < params.MAX_GLOBAL_STEPS:
            frame_idx += params.TRAIN_STEP_FREQ
            buffer.populate_stacked_experience(params.TRAIN_STEP_FREQ)
            epsilon_tracker.udpate(frame_idx)

            episode_rewards = exp_source.pop_episode_reward_lst()
            if episode_rewards:
                solved, mean_episode_reward = reward_tracker.set_episode_reward(
                    episode_rewards[0], frame_idx, action_selector.epsilon, action_count
                )

                if frame_idx >= next_save_frame_idx:
                    dqn_model.save_model(
                        MODEL_SAVE_DIR, params.ENVIRONMENT_ID.value, net.__name__, net, frame_idx, mean_episode_reward
                    )
                    next_save_frame_idx += params.MODEL_SAVE_STEP_PERIOD

                if solved:
                    dqn_model.save_model(
                        MODEL_SAVE_DIR, params.ENVIRONMENT_ID.value, net.__name__, net, frame_idx, mean_episode_reward
                    )
                    break

            if len(buffer) < params.MIN_REPLAY_SIZE_FOR_TRAIN:
                continue

            optimizer.zero_grad()
            batch = buffer.sample(params.BATCH_SIZE)
            loss_v = dqn_model.calc_loss_dqn(batch, net, tgt_net, gamma=params.GAMMA, cuda=params.CUDA)
            loss_v.backward()
            optimizer.step()

            if params.DRAW_VIZ and frame_idx % 1000 == 0:
                stat_for_model_loss.draw_optimization_performance(frame_idx, loss_v.item())

            if frame_idx % params.TARGET_NET_SYNC_STEP_PERIOD < params.TRAIN_STEP_FREQ:
                tgt_net.sync()