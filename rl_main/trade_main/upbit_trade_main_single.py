#!/usr/bin/env python3
import copy

import torch
import torch.optim as optim
import os
import warnings

from common import common_utils
from common.environments.trade.trade_constant import TimeUnit, EnvironmentType
from common.environments.trade.trade_env import UpbitEnvironment

from common.fast_rl import rl_agent, value_based_model, actions, experience_single, replay_buffer
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


def test(test_env, net):
    action_selector = actions.ArgmaxActionSelector()
    agent = rl_agent.DQNAgent(net, action_selector, device=device)

    experience_source = experience_single.ExperienceSourceSingleEnvFirstLast(test_env, agent, gamma=params.GAMMA, steps_count=params.N_STEP)

    done = False
    state = test_env.reset()
    agent_state = agent.initial_agent_state()

    episode_reward = 0.0
    info = None

    while not done:
        states_input = []
        processed_state = experience_source.get_processed_state(state)
        states_input.append(processed_state)

        agent_states_input = []
        agent_states_input.append(agent_state)

        new_actions, new_agent_states = agent(states_input, agent_states_input)

        agent_state = new_agent_states[0]
        action = new_actions[0]

        next_state, reward, done, info = test_env.step(action)

        episode_reward += reward
        state = next_state

    return episode_reward, info["profit"]


def main():
    common_utils.print_fast_rl_params(params)

    params.BATCH_SIZE *= params.TRAIN_STEP_FREQ

    env = UpbitEnvironment(
        coin_name="MOC", time_unit=TimeUnit.ONE_HOUR, environment_type=EnvironmentType.TRAIN
    )

    test_env = copy.deepcopy(env)

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

    experience_source = experience_single.ExperienceSourceSingleEnvFirstLast(env, agent, gamma=params.GAMMA, steps_count=params.N_STEP)
    buffer = replay_buffer.ExperienceReplayBuffer(experience_source, buffer_size=params.REPLAY_BUFFER_SIZE)
    optimizer = optim.Adam(net.parameters(), lr=params.LEARNING_RATE)

    if params.DRAW_VIZ:
        stat = statistics.StatisticsForValueBasedRL(method="nature_dqn")
        stat_for_model_loss = statistics.StatisticsForValueBasedOptimization()
    else:
        stat = None
        stat_for_model_loss = None

    step_idx = 0

    last_loss = 0.0

    with utils.RewardTracker(params=params, frame=False, stat=stat) as reward_tracker:
        while step_idx < params.MAX_GLOBAL_STEPS:
            step_idx += params.TRAIN_STEP_FREQ
            last_entry = buffer.populate(params.TRAIN_STEP_FREQ)

            epsilon_tracker.udpate(step_idx)

            episode_rewards = experience_source.pop_episode_reward_lst()

            if episode_rewards:
                for episode_reward in episode_rewards:
                    solved, mean_episode_reward = reward_tracker.set_episode_reward(
                        episode_reward, step_idx, action_selector.epsilon, last_info=last_entry.info, last_loss=last_loss
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

                    if reward_tracker.done_episodes % params.TEST_EPISODE_PERIOD == 0:
                        episode_reward, profit = test(test_env, net)
                        print("[TEST] episode_reward: {0:7.3f}, profit: {1:6.1f}".format(episode_reward, profit))

            if len(buffer) < params.MIN_REPLAY_SIZE_FOR_TRAIN:
                continue

            optimizer.zero_grad()
            batch = buffer.sample(params.BATCH_SIZE)
            loss_v = value_based_model.calc_loss_dqn(batch, net, tgt_net, gamma=params.GAMMA, cuda=params.CUDA)
            loss_v.backward()
            optimizer.step()

            draw_loss = min(1.0, loss_v.detach().item())
            last_loss = loss_v.detach().item()

            if params.DRAW_VIZ and step_idx % 1000 == 0:
                stat_for_model_loss.draw_optimization_performance(step_idx, draw_loss)

            if step_idx % params.TARGET_NET_SYNC_STEP_PERIOD < params.TRAIN_STEP_FREQ:
                tgt_net.sync()




if __name__ == "__main__":
    main()