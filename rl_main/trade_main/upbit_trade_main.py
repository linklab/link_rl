#!/usr/bin/env python3
import copy

import torch
import torch.optim as optim
import os, sys
import warnings

current_path = os.path.dirname(os.path.realpath(__file__))
PROJECT_HOME = os.path.abspath(os.path.join(current_path, os.pardir, os.pardir))
if PROJECT_HOME not in sys.path:
    sys.path.append(PROJECT_HOME)

from common.fast_rl.common.utils import EarlyStopping
from common.environments.trade.trade_data import get_data
from common import common_utils
from common.environments.trade.trade_constant import TimeUnit, EnvironmentType, Action, WINDOW_SIZE
from common.environments.trade.trade_env import UpbitEnvironment, EpsilonGreedyTradeDQNActionSelector, \
    ArgmaxTradeActionSelector

from common.fast_rl import rl_agent, value_based_model, actions, experience_single, replay_buffer
from common.fast_rl.common import utils
from common.fast_rl.common import statistics

##### NOTE #####
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


def test(env, net, verbose=True):
    action_selector = ArgmaxTradeActionSelector(env=env)
    agent = rl_agent.DQNAgent(net, action_selector, device=device)
    experience_source = experience_single.ExperienceSourceSingleEnvFirstLast(env, agent, gamma=params.GAMMA, steps_count=params.N_STEP)

    done = False
    state = env.reset()
    agent_state = agent.initial_agent_state()

    episode_reward = 0.0
    num_buys = 0
    info = None
    step_idx = 0
    while not done:
        step_idx += 1
        states_input = []
        processed_state = experience_source.get_processed_state(state)
        states_input.append(processed_state)

        agent_states_input = []
        agent_states_input.append(agent_state)

        new_actions, new_agent_states = agent(states_input, agent_states_input)

        agent_state = new_agent_states[0]
        action = new_actions[0]

        if action == Action.MARKET_BUY.value:
            num_buys += 1
            if num_buys > 10:
                action_str = "BUY({0})".format(10)
            else:
                action_str = "BUY({0})".format(num_buys)
        else:
            action_str = env.get_action_meanings()[action]

        msg = "[{0:2}|{1}] OHLCV: {2}, {3}, {4}, {5}, {6:<10.1f}, Action: {7:7} --> ".format(
            step_idx,
            env.data.iloc[env.transaction_state_idx]['datetime_krw'],
            env.data.iloc[env.transaction_state_idx]['open'],
            env.data.iloc[env.transaction_state_idx]['high'],
            env.data.iloc[env.transaction_state_idx]['low'],
            env.data.iloc[env.transaction_state_idx]['final'],
            env.data.iloc[env.transaction_state_idx]['volume'],
            action_str
        )

        next_state, reward, done, info = env.step(action)

        if action in [Action.HOLD.value]:
            msg += "Reward: {0:.3f}, hold coin: {1:.1f}".format(
                reward, info["hold_coin"]
            )
        elif action == Action.MARKET_BUY.value:
            if num_buys <= 10:
                coin_krw_str = "{0:.1f}".format(info['coin_krw'])
                commission_fee_str = "{0:.1f}".format(info['commission_fee'])
            else:
                coin_krw_str = "-"
                commission_fee_str = "-"

            msg += "Reward: {0:.3f}, slippage: {1:.1f}, coin_unit_price: {2:.1f}, " \
                   "coin_krw: {3}, commission: {4}, hold coin: {5:.1f}".format(
                reward, info["slippage"], info["coin_unit_price"],
                coin_krw_str, commission_fee_str, info["hold_coin"]
            )
        elif action == Action.MARKET_SELL.value:
            msg += "Reward: {0:.3f}, slippage: {1:.1f}, coin_unit_price: {2:.1f}, " \
                   "coin_krw: {3:.1f}, commission: {4:.1f}, sold coin: {5:.1f}, profit: {6:.1f}".format(
                reward, info["slippage"], info["coin_unit_price"],
                info['coin_krw'], info['commission_fee'], info["sold_coin"], info["profit"]
            )
        else:
            raise ValueError()
        if verbose:
            print(msg)

        episode_reward += reward
        state = next_state

    print("TRANSACTION START DATETIME: {0}, EPISODE REWARD: {1:>8.3f}, PROFIT: {2:>10.1f}, STEPS: {3}".format(
        env.transaction_start_datetime, episode_reward, info["profit"], step_idx
    ))

    return info["profit"], step_idx


def train(train_env, test_env):
    common_utils.print_fast_rl_params(params)

    params.BATCH_SIZE *= params.TRAIN_STEP_FREQ

    net = value_based_model.DuelingDQNSmallCNN(
        input_shape=train_env.observation_space.shape,
        n_actions=train_env.action_space.n
    ).to(device)
    print(net)
    print("ACTION MEANING: {0}".format(train_env.get_action_meanings()))

    tgt_net = rl_agent.TargetNet(net)

    action_selector = EpsilonGreedyTradeDQNActionSelector(epsilon=params.EPSILON_INIT, env=train_env)
    epsilon_tracker = actions.EpsilonTracker(
        action_selector=action_selector,
        eps_start=params.EPSILON_INIT,
        eps_final=params.EPSILON_MIN,
        eps_frames=params.EPSILON_MIN_STEP
    )
    agent = rl_agent.DQNAgent(net, action_selector, device=device)

    experience_source = experience_single.ExperienceSourceSingleEnvFirstLast(train_env, agent, gamma=params.GAMMA, steps_count=params.N_STEP)
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

    early_stopping = EarlyStopping(
        patience=7,
        stop_mean_episode_reward=params.STOP_MEAN_EPISODE_REWARD,
        verbose=True,
        delta=0.0,
        model_save_dir=MODEL_SAVE_DIR,
        env_name=params.ENVIRONMENT_ID.value,
        model_name=net.__name__
    )

    with utils.RewardTracker(params=params, frame=False, stat=stat, early_stopping=early_stopping) as reward_tracker:
        while step_idx < params.MAX_GLOBAL_STEP:
            step_idx += params.TRAIN_STEP_FREQ
            last_entry = buffer.populate(params.TRAIN_STEP_FREQ)

            epsilon_tracker.udpate(step_idx)

            episode_rewards = experience_source.pop_episode_reward_lst()

            if episode_rewards:
                for episode_reward in episode_rewards:
                    solved, _ = reward_tracker.set_episode_reward(
                        episode_reward, step_idx, action_selector.epsilon, last_info=last_entry.info,
                        last_loss=last_loss, model=net
                    )

                    if solved:
                        break

                    if reward_tracker.done_episodes % params.TEST_PERIOD_EPISODE == 0:
                        print("#" * 200)
                        print("[TEST START]")
                        test(test_env, net)
                        print("[TEST END]")
                        print("#" * 200)

                if solved:
                    break

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
    return net


def test_random(env, net, num_episodes):
    num_positive = 0
    num_negative = 0
    total_profit = 0.0
    total_steps = 0

    for _ in range(num_episodes):
        profit, step = test(env, net, verbose=False)
        if profit > 0:
            num_positive += 1
        else:
            num_negative += 1
        total_profit += profit
        total_steps += step

    print("### POSTITIVE: {0}/{2}, NEGATIVE: {1}/{2}, TOTAL PROFIT: {3:.1f}, AVG. STEP FOR EPISODE: {4:.1f}".format(
        num_positive, num_negative, num_episodes, total_profit, total_steps / num_episodes
    ))


def test_sequential_all(env, net, data_size):
    num_positive = 0
    num_negative = 0
    total_profit = 0.0
    total_steps = 0

    num_episodes = 0
    env.transaction_state_idx = 0
    while True:
        num_episodes += 1
        profit, step = test(env, net, verbose=False)
        if profit > 0:
            num_positive += 1
        else:
            num_negative += 1
        total_profit += profit
        total_steps += step

        if env.transaction_state_idx >= data_size:
            break

    print("### POSITIVE: {0}/{2}, NEGATIVE: {1}/{2}, TOTAL PROFIT: {3:.1f}, AVG. STEP FOR EPISODE: {4:.1f}".format(
        num_positive, num_negative, num_episodes, total_profit, total_steps / num_episodes
    ))


if __name__ == "__main__":
    coin_name = "OMG"
    time_unit = TimeUnit.ONE_DAY

    train_data_info, test_data_info = get_data(coin_name=coin_name, time_unit=time_unit)

    print(train_data_info["first_datetime_krw"], train_data_info["last_datetime_krw"])
    print(test_data_info["first_datetime_krw"], test_data_info["last_datetime_krw"])

    train_env = UpbitEnvironment(
        coin_name=coin_name,
        time_unit=time_unit,
        data_info=train_data_info,
        environment_type=EnvironmentType.TRAIN
    )

    test_random_env = UpbitEnvironment(
        coin_name=coin_name,
        time_unit=time_unit,
        data_info=test_data_info,
        environment_type=EnvironmentType.TEST_RANDOM,
    )

    net = train(train_env, test_random_env)

    # print("#### TEST RANDOM 100")
    # test_random(test_random_env, net, num_episodes=100)

    print()

    print("#### TEST SEQUENTIALLY")
    test_sequential_env = UpbitEnvironment(
        coin_name=coin_name,
        time_unit=time_unit,
        data_info=test_data_info,
        environment_type=EnvironmentType.TEST_SEQUENTIAL,
    )
    test_sequential_all(test_sequential_env, net, len(test_data_info["data"]))
