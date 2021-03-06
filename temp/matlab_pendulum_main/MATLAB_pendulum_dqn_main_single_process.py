#!/usr/bin/env python3
import sys

from codes.e_utils import rl_utils
from codes.e_utils.names import PROJECT_HOME

sys.path.insert(0,"c:\\users\\wlckd\\anaconda3\\envs\\link_rl\\lib\\site-packages")
import torch
from torch import optim
import os

from common.environments import MatlabRotaryInvertedPendulumEnv
print(torch.__version__)

from common.fast_rl import actions, value_based_model, rl_agent, experience_single, replay_buffer
from common.fast_rl.common import statistics, utils
from codes.a_config.parameters import PARAMETERS as params

cuda = False
# env_name = 'CartPole-v1'


MODEL_SAVE_DIR = os.path.join(PROJECT_HOME, "out", "model_save_files")
if not os.path.exists(MODEL_SAVE_DIR):
    os.makedirs(MODEL_SAVE_DIR)


os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


def main():
    env = rl_utils.get_environment(owner="worker", params=params)

    net = value_based_model.DuelingDQNMLP(
        obs_size=4,
        hidden_size_1=128, hidden_size_2=128,
        n_actions=7
    ).to(device)

    tgt_net = rl_agent.TargetNet(net)

    buffer = replay_buffer.PrioReplayBuffer(experience_source=None, buffer_size=params.REPLAY_BUFFER_SIZE)
    optimizer = optim.Adam(net.parameters(), lr=params.LEARNING_RATE)

    stat_for_value_based_rl = statistics.StatisticsForValueBasedRL(method="nature_dqn")
    stat_for_model_loss = statistics.StatisticsForValueBasedOptimization()

    action_selector = actions.EpsilonGreedyDQNActionSelector(epsilon=params.EPSILON_INIT)

    epsilon_tracker = actions.EpsilonTracker(
        action_selector=action_selector,
        eps_start=params.EPSILON_INIT,
        eps_final=params.EPSILON_MIN,
        eps_frames=params.EPSILON_MIN_STEP
    )

    agent = rl_agent.DQNAgent(net, action_selector, device=device)
    experience_source = experience_single.ExperienceSourceSingleEnvFirstLast(
        env, agent, params.GAMMA, steps_count=params.N_STEP
    )
    exp_source_iter = iter(experience_source)

    next_save_frame_idx = params.MODEL_SAVE_STEP_PERIOD
    frame_idx = 0

    with utils.RewardTracker(params=params, frame=False, stat=stat_for_value_based_rl) as reward_tracker:
        while frame_idx < params.MAX_GLOBAL_STEP:
            frame_idx += 1
            exp = next(exp_source_iter)

            epsilon_tracker.udpate(frame_idx)

            buffer._add(exp)

            if frame_idx % params.TRAIN_STEP_FREQ == 0:
                model_update(frame_idx, buffer, stat_for_model_loss, optimizer, net, tgt_net)

            episode_rewards = experience_source.pop_episode_reward_lst()
            if episode_rewards:
                solved, mean_episode_reward = reward_tracker.set_episode_reward(
                    episode_rewards[0], frame_idx, action_selector.epsilon
                )

                if frame_idx >= next_save_frame_idx:
                    rl_agent.save_model(MODEL_SAVE_DIR, params.ENVIRONMENT_ID, net.__name__, net, frame_idx, mean_episode_reward)
                    next_save_frame_idx += params.MODEL_SAVE_STEP_PERIOD

                if solved:
                    rl_agent.save_model(MODEL_SAVE_DIR, params.ENVIRONMENT_ID, net.__name__, net, frame_idx, mean_episode_reward)
                    break



def model_update(frame_idx, buffer, stat_for_model_loss, optimizer, net, tgt_net):
    if len(buffer) < params.MIN_REPLAY_SIZE_FOR_TRAIN:
        if params.DRAW_VIZ and frame_idx % 100 == 0:
            stat_for_model_loss.draw_optimization_performance(frame_idx, 0.0)
        return

    optimizer.zero_grad()
    batch, batch_indices, batch_weights = buffer.sample(params.BATCH_SIZE)
    loss_v, sample_prios = value_based_model.calc_loss_per_double_dqn(
        buffer.buffer, batch, batch_indices, batch_weights, net, tgt_net, params, cuda=cuda, cuda_async=True
    )
    loss_v.backward()
    optimizer.step()
    buffer.update_priorities(batch_indices, sample_prios.data.cpu().numpy())
    buffer.update_beta(frame_idx)

    if params.DRAW_VIZ and frame_idx % 100 == 0:
        stat_for_model_loss.draw_optimization_performance(frame_idx, loss_v.item())

    if frame_idx % params.TARGET_NET_SYNC_STEP_PERIOD < params.TRAIN_STEP_FREQ:
        tgt_net.sync()


if __name__ == "__main__":
    main()
