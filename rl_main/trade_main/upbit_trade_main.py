#!/usr/bin/env python3
import time
import torch
import torch.optim as optim
import torch.multiprocessing as mp
import os
import warnings
from collections import deque

from common.environments.trade.trade_constant import TimeUnit, EnvironmentType, Action
from common.environments.trade.trade_env import UpbitEnvironment

from common import common_utils
from common.fast_rl import experience, rl_agent, value_based_model, actions
from common.fast_rl.common import utils
from common.fast_rl.common import statistics
from config.names import PROJECT_HOME

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
    experience_source = experience.ExperienceSourceSingleEnvFirstLast(
        env=env, agent=agent, gamma=params.GAMMA, steps_count=params.N_STEP
    )
    exp_source_iter = iter(experience_source)

    if params.DRAW_VIZ:
        stat = statistics.StatisticsForValueBasedRL(method="nature_dqn")
    else:
        stat = None

    action_count = []
    for _ in env.get_action_meanings():
        action_count.append(0)

    step_idx = 0

    with utils.RewardTracker(params=params, frame=False, stat=stat) as reward_tracker:
        while step_idx < params.MAX_GLOBAL_STEPS:
            step_idx += 1
            exp = next(exp_source_iter)
            action_count[exp.action] += 1
            exp_queue.put(exp)

            epsilon_tracker.udpate(step_idx)

            episode_rewards = experience_source.pop_episode_reward_lst()

            if episode_rewards:

                current_episode_reward = episode_rewards[0]
                
                solved, mean_episode_reward = reward_tracker.set_episode_reward(
                    current_episode_reward, step_idx, action_selector.epsilon, action_count
                )

                if solved:
                    rl_agent.save_model(
                        MODEL_SAVE_DIR,
                        params.ENVIRONMENT_ID.value,
                        net.__name__,
                        net,
                        step_idx,
                        mean_episode_reward
                    )
                    break

    exp_queue.put(None)


def main():
    mp.set_start_method('spawn')

    common_utils.print_fast_rl_params(params)

    params.BATCH_SIZE *= params.TRAIN_STEP_FREQ

    env = UpbitEnvironment(
        coin_name="MOC", time_unit=TimeUnit.ONE_HOUR, environment_type=EnvironmentType.TRAIN
    )

    net = value_based_model.DuelingDQNSmallCNN(
        input_shape=env.observation_space.shape,
        n_actions=env.action_space.n
    ).to(device)
    print(net)
    print("ACTION MEANING: {0}".format(env.get_action_meanings()))

    tgt_net = rl_agent.TargetNet(net)

    buffer = experience.PrioritizedReplayBuffer(experience_source=None, buffer_size=params.REPLAY_BUFFER_SIZE, n_step=params.N_STEP)
    optimizer = optim.Adam(net.parameters(), lr=params.LEARNING_RATE)

    exp_queue = mp.Queue(maxsize=params.TRAIN_STEP_FREQ * 2)
    play_proc = mp.Process(target=play_func, args=(env, net, exp_queue))
    play_proc.start()

    time.sleep(0.5)
    if params.DRAW_VIZ:
        stat_for_model_loss = statistics.StatisticsForValueBasedOptimization()
    else:
        stat_for_model_loss = None

    loss_list = deque(maxlen=params.AVG_EPISODE_SIZE_FOR_STAT)

    step_idx = 0

    while play_proc.is_alive():
        step_idx += params.TRAIN_STEP_FREQ
        for _ in range(params.TRAIN_STEP_FREQ):
            exp = exp_queue.get()
            if exp is None:
                play_proc.join()
                break
            buffer._add(exp)

        if len(buffer) < params.MIN_REPLAY_SIZE_FOR_TRAIN:
            if params.DRAW_VIZ and step_idx % 1000 == 0:
                stat_for_model_loss.draw_optimization_performance(step_idx, 0.0)
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
        buffer.update_beta(step_idx)

        draw_loss = min(10.0, loss_v.detach().item())

        if params.DRAW_VIZ and step_idx % 1000 == 0:
            stat_for_model_loss.draw_optimization_performance(step_idx, draw_loss)

        if step_idx % params.TARGET_NET_SYNC_STEP_PERIOD < params.TRAIN_STEP_FREQ:
            tgt_net.sync()

        loss_list.append(loss_v.detach().item())


if __name__ == "__main__":
    main()