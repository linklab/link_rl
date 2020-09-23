#!/usr/bin/env python3
import time
import numpy as np
import gym
import torch
import torch.optim as optim
import torch.multiprocessing as mp
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


def play_func(env, net, exp_queue):
    action_selector = actions.EpsilonGreedyActionSelector(epsilon=params.EPSILON_INIT)
    epsilon_tracker = actions.EpsilonTracker(
        action_selector=action_selector,
        eps_start=params.EPSILON_INIT,
        eps_final=params.EPSILON_MIN,
        eps_frames=params.EPSILON_MIN_STEP
    )
    agent = rl_agent.DQNAgent(net, action_selector, device=device)
    if params.OMEGA:
        exp_source = experience.ExperienceSource(env, agent, steps_count=1)
    else:
        exp_source = experience.ExperienceSourceFirstLast(env, agent, gamma=params.GAMMA, steps_count=params.N_STEP)
    exp_source_iter = iter(exp_source)
    stat = statistics.Statistics(method="nature_dqn")

    action_count = []
    for _ in env.unwrapped.get_action_meanings():
        action_count.append(0)

    frame_idx = 0
    next_save_frame_idx = params.MODEL_SAVE_STEP_PERIOD

    with utils.AtariRewardTracker(stop_mean_episode_reward=params.STOP_MEAN_EPISODE_REWARD,
            average_size_for_stats=params.AVG_EPISODE_SIZE_FOR_STAT,
            draw_viz=params.DRAW_VIZ, stat=stat) as reward_tracker:
        while frame_idx < params.MAX_GLOBAL_STEPS:
            frame_idx += 1
            exp = next(exp_source_iter)
            action_count[exp.action] += 1
            exp_queue.put(exp)

            epsilon_tracker.udpate(frame_idx)

            episode_rewards = exp_source.pop_episode_reward_lst()
            if episode_rewards:
                solved, mean_episode_reward = reward_tracker.reward(episode_rewards[0], frame_idx, action_selector.epsilon, action_count)

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

    exp_queue.put(None)


def main():
    mp.set_start_method('spawn')

    common_utils.print_fast_rl_params(params)

    if params.SEED is not None:
        np.random.seed(params.SEED)
        torch.manual_seed(params.SEED)
        torch.cuda.manual_seed_all(params.SEED)

    params.BATCH_SIZE *= params.TRAIN_STEP_FREQ

    env = make_atari_env(params)
    if params.SEED is not None:
        env.seed(params.SEED)
    suffix = "" if params.SEED is None else "_seed=%s" % params.SEED

    net = dqn_model.DuelingDQNCNN(
        input_shape=env.observation_space.shape,
        n_actions=env.action_space.n
    ).to(device)
    print(net)
    print("ACTION MEANING: {0}".format(env.unwrapped.get_action_meanings()))

    tgt_net = rl_agent.TargetNet(net)

    buffer = experience.PrioReplayBuffer(exp_source=None, buf_size=params.REPLAY_BUFFER_SIZE)
    optimizer = optim.Adam(net.parameters(), lr=params.LEARNING_RATE)

    exp_queue = mp.Queue(maxsize=params.TRAIN_STEP_FREQ * 2)
    play_proc = mp.Process(target=play_func, args=(env, net, exp_queue))
    play_proc.start()

    time.sleep(0.5)
    stat_for_model_loss = statistics.StatisticsForModelLoss()
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
            if params.DRAW_VIZ and frame_idx % 1000 == 0:
                stat_for_model_loss.draw_loss(frame_idx, 0.0)
            continue

        optimizer.zero_grad()
        batch, batch_indices, batch_weights = buffer.sample(params.BATCH_SIZE)
        if params.OMEGA:
            loss_v, sample_prios =dqn_model.calc_loss_per_double_dqn_for_omega(
                buffer.buffer, batch, batch_weights, net, tgt_net, params, cuda=params.cuda, cuda_async=True
            )
        else:
            loss_v, sample_prios = dqn_model.calc_loss_per_double_dqn(
                buffer.buffer, batch, batch_weights, net, tgt_net, gamma=params.GAMMA, cuda=params.CUDA, cuda_async=True
            )
        loss_v.backward()
        optimizer.step()
        buffer.update_priorities(batch_indices, sample_prios)
        buffer.update_beta(frame_idx)

        if params.DRAW_VIZ and frame_idx % 1000 == 0:
            stat_for_model_loss.draw_loss(frame_idx, loss_v.item())

        if frame_idx % params.TARGET_NET_SYNC_STEP_PERIOD < params.TRAIN_STEP_FREQ:
            tgt_net.sync()

# python atari_dqn.py --env=pong --draw_viz=1 --cuda
# python atari_dqn.py --env=breakout --draw_viz=1 --cuda
if __name__ == "__main__":
    main()
