#!/usr/bin/env python3
import torch
import torch.optim as optim
import os
import warnings

from common import common_utils
from common.common_utils import make_atari_env
from common.environments.trade.trade_constant import TimeUnit, EnvironmentType
from common.environments.trade.trade_env import UpbitEnvironment
from common.fast_rl import experience, rl_agent, value_based_model, actions
from common.fast_rl.common import utils
from common.fast_rl.common import statistics, wrappers

##### NOTE #####
from config.names import PROJECT_HOME
from config.parameters import PARAMETERS as params
##### NOTE #####

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

if torch.cuda.is_available():
    device = torch.device("cuda" if params.CUDA else "cpu")
else:
    device = torch.device("cpu")

MODEL_SAVE_DIR = os.path.join(PROJECT_HOME, "out", "model_save_files")
if not os.path.exists(MODEL_SAVE_DIR):
    os.makedirs(MODEL_SAVE_DIR)


if __name__ == "__main__":
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

    action_selector = actions.EpsilonGreedyDQNActionSelector(epsilon=params.EPSILON_INIT)
    epsilon_tracker = actions.EpsilonTracker(
        action_selector=action_selector,
        eps_start=params.EPSILON_INIT,
        eps_final=params.EPSILON_MIN,
        eps_frames=params.EPSILON_MIN_STEP
    )
    agent = rl_agent.DQNAgent(net, action_selector, device=device)

    experience_source = experience.ExperienceSourceSingleEnvFirstLast(env, agent, gamma=params.GAMMA, steps_count=params.N_STEP)
    buffer = experience.ExperienceReplayBuffer(experience_source, buffer_size=params.REPLAY_BUFFER_SIZE)
    optimizer = optim.Adam(net.parameters(), lr=params.LEARNING_RATE)

    if params.DRAW_VIZ:
        stat = statistics.StatisticsForValueBasedRL(method="nature_dqn")
        stat_for_model_loss = statistics.StatisticsForValueBasedOptimization()
    else:
        stat = None
        stat_for_model_loss = None

    action_count = []
    for _ in env.get_action_meanings():
        action_count.append(0)

    step_idx = 0

    with utils.RewardTracker(params=params, frame=False, stat=stat) as reward_tracker:
        while step_idx < params.MAX_GLOBAL_STEPS:
            step_idx += params.TRAIN_STEP_FREQ
            buffer.populate_with_action_count(params.TRAIN_STEP_FREQ, action_count)

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

            if len(buffer) < params.MIN_REPLAY_SIZE_FOR_TRAIN:
                continue

            optimizer.zero_grad()
            batch = buffer.sample(params.BATCH_SIZE)
            loss_v = value_based_model.calc_loss_dqn(batch, net, tgt_net, gamma=params.GAMMA, cuda=params.CUDA)
            loss_v.backward()
            optimizer.step()

            draw_loss = min(1.0, loss_v.detach().item())

            if params.DRAW_VIZ and step_idx % 1000 == 0:
                stat_for_model_loss.draw_optimization_performance(step_idx, draw_loss)

            if step_idx % params.TARGET_NET_SYNC_STEP_PERIOD < params.TRAIN_STEP_FREQ:
                tgt_net.sync()