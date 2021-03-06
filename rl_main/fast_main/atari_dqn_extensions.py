#!/usr/bin/env python3
import time
import torch
import torch.optim as optim
import torch.multiprocessing as mp
import numpy as np
import os
import warnings
from collections import deque

from codes.e_utils import common_utils
from codes.e_utils.common_utils import make_atari_env
from common.fast_rl import experience, rl_agent, value_based_model, actions, replay_buffer
from common.fast_rl.common import utils
from common.fast_rl.common import statistics
from config.names import PROJECT_HOME
from rl_main.fast_main.atari_draw_graph import save_reward_as_pickle, save_q_loss_as_pickle

# from line_profiler import LineProfiler
# from memory_profiler import profile

##### NOTE #####
from config.parameters import PARAMETERS as params
##### NOTE #####

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['LRU_CACHE_CAPACITY'] = '1'

if torch.cuda.is_available():
    device = torch.device("cuda" if params.CUDA else "cpu")
else:
    device = torch.device("cpu")

MODEL_SAVE_DIR = os.path.join(PROJECT_HOME, "out", "model_save_files")
if not os.path.exists(MODEL_SAVE_DIR):
    os.makedirs(MODEL_SAVE_DIR)


def play_func(env, net, exp_queue):
    action_selector = actions.EpsilonGreedyDQNActionSelector(epsilon=params.EPSILON_INIT)
    epsilon_tracker = actions.EpsilonTracker(
        action_selector=action_selector,
        eps_start=params.EPSILON_INIT,
        eps_final=params.EPSILON_MIN,
        eps_frames=params.EPSILON_MIN_STEP
    )
    agent = rl_agent.DQNAgent(net, action_selector, device=device)
    experience_source = experience.ExperienceSourceNamedTuple(env, agent, steps_count=1)
    exp_source_iter = iter(experience_source)

    if params.DRAW_VIZ:
        stat = statistics.StatisticsForValueBasedRL(method="nature_dqn")
    else:
        stat = None

    episode_rewards_across_steps = np.zeros(int(params.MAX_GLOBAL_STEP / params.DATA_SAVE_STEP_PERIOD))
    last_mean_episode_reward = 0

    frame_idx = 0
    next_save_frame_idx = params.MODEL_SAVE_STEP_PERIOD

    with utils.RewardTracker(params=params, frame=True, stat=stat) as reward_tracker:
        while frame_idx < params.MAX_GLOBAL_STEP:
            frame_idx += 1
            exp = next(exp_source_iter)
            exp_queue.put(exp)

            epsilon_tracker.udpate(frame_idx)

            if frame_idx % params.DATA_SAVE_STEP_PERIOD == 0:
                episode_rewards_across_steps[int((frame_idx-1)/params.DATA_SAVE_STEP_PERIOD)] = last_mean_episode_reward
                save_reward_as_pickle(episode_rewards_across_steps, params)

            episode_rewards = experience_source.pop_episode_reward_lst()
            if episode_rewards:
                solved, mean_episode_reward = reward_tracker.set_episode_reward(
                    episode_rewards[0], frame_idx, action_selector.epsilon
                )
                last_mean_episode_reward = mean_episode_reward

                if frame_idx >= next_save_frame_idx:
                    rl_agent.save_model(
                        MODEL_SAVE_DIR, params.ENVIRONMENT_ID.value, net.__name__, net, frame_idx, mean_episode_reward
                    )
                    next_save_frame_idx += params.MODEL_SAVE_STEP_PERIOD

                if solved:
                    rl_agent.save_model(
                        MODEL_SAVE_DIR, params.ENVIRONMENT_ID.value, net.__name__, net, frame_idx, mean_episode_reward
                    )
                    break

    exp_queue.put(None)


def main():
    mp.set_start_method('spawn')

    common_utils.print_fast_rl_params(params)

    params.BATCH_SIZE *= params.TRAIN_STEP_FREQ

    env = make_atari_env(params.ENVIRONMENT_ID.value, seed=params.SEED)
    if params.SEED is not None:
        env.seed(params.SEED)
    suffix = "" if params.SEED is None else "_seed=%s" % params.SEED

    net = value_based_model.DuelingDQNCNN(
        input_shape=env.observation_space.shape,
        n_actions=env.action_space.n
    ).to(device)
    print(net)
    print("ACTION MEANING: {0}".format(env.unwrapped.get_action_meanings()))

    tgt_net = rl_agent.TargetNet(net)

    if params.OMEGA:
        # buffer = replay_buffer.PrioReplayBuffer(experience_source=None, buffer_size=params.REPLAY_BUFFER_SIZE, n_step=params.OMEGA_WINDOW_SIZE)
        buffer = replay_buffer.PrioritizedReplayBuffer(experience_source=None, buffer_size=params.REPLAY_BUFFER_SIZE, n_step=params.OMEGA_WINDOW_SIZE)
    else:
        # buffer = replay_buffer.PrioReplayBuffer(experience_source=None, buffer_size=params.REPLAY_BUFFER_SIZE, n_step=params.N_STEP)
        buffer = replay_buffer.PrioritizedReplayBuffer(experience_source=None, buffer_size=params.REPLAY_BUFFER_SIZE, n_step=params.N_STEP)
    optimizer = optim.Adam(net.parameters(), lr=params.LEARNING_RATE)

    exp_queue = mp.Queue(maxsize=params.TRAIN_STEP_FREQ * 2)
    play_proc = mp.Process(target=play_func, args=(env, net, exp_queue))
    play_proc.start()

    time.sleep(0.5)
    if params.DRAW_VIZ:
        stat_for_model_loss = statistics.StatisticsForValueBasedOptimization()
    else:
        stat_for_model_loss = None

    q_loss_across_steps = np.zeros(int(params.MAX_GLOBAL_STEP / params.DATA_SAVE_STEP_PERIOD))
    loss_list = deque(maxlen=params.AVG_EPISODE_SIZE_FOR_STAT)

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
                stat_for_model_loss.draw_optimization_performance(frame_idx, 0.0)
            continue

        optimizer.zero_grad()
        batch, batch_indices, batch_weights = buffer.sample(params.BATCH_SIZE)
        if params.OMEGA:
            loss_v, sample_prios =value_based_model.calc_loss_per_double_dqn_for_omega(
                buffer.buffer, batch, batch_indices, batch_weights, net, tgt_net, params, cuda=params.CUDA, cuda_async=True
            )
        else:
            loss_v, sample_prios = value_based_model.calc_loss_per_double_dqn(
                buffer.buffer, batch, batch_indices, batch_weights, net, tgt_net, params, cuda=params.CUDA, cuda_async=True
            )
        loss_v.backward()

        optimizer.step()
        buffer.update_priorities(batch_indices, sample_prios.detach().cpu().numpy())       # .detach().data.cpu().numpy()
        buffer.update_beta(frame_idx)

        if params.DRAW_VIZ and frame_idx % 1000 == 0:
            stat_for_model_loss.draw_optimization_performance(frame_idx, loss_v.detach().item())

        if frame_idx % params.TARGET_NET_SYNC_STEP_PERIOD < params.TRAIN_STEP_FREQ:
            tgt_net.sync()

        loss_list.append(loss_v.detach().item())
        if frame_idx % params.DATA_SAVE_STEP_PERIOD < params.TRAIN_STEP_FREQ:
            q_loss_across_steps[int((frame_idx - 1) / params.DATA_SAVE_STEP_PERIOD)] = np.mean(loss_list)
            save_q_loss_as_pickle(q_loss_across_steps, params)
            # gc.collect()

        # del loss_v
        # del loss_v, sample_prios
        # gc.collect()

        # if frame_idx % 10000 == 0:
        #     lp.print_stats()


# python atari_dqn.py --env=pong --draw_viz=1 --cuda
# python atari_dqn.py --env=breakout --draw_viz=1 --cuda
if __name__ == "__main__":
    # lp = LineProfiler()
    # lp_wrapper = lp(main)
    # lp_wrapper(lp)
    main()
