#!/usr/bin/env python3
import time
import torch
import os, sys
import numpy as np

current_path = os.path.dirname(os.path.realpath(__file__))
PROJECT_HOME = os.path.abspath(os.path.join(current_path, os.pardir, os.pardir, os.pardir, os.pardir))
if PROJECT_HOME not in sys.path:
    sys.path.append(PROJECT_HOME)

from codes.a_config.parameters import PARAMETERS as params

from codes.e_utils.common_utils import load_model
from codes.e_utils import rl_utils
from codes.e_utils.actions import EpsilonGreedySomeTimesBlowDQNActionSelector, \
    EpsilonGreedySomeTimesBlowDDPGActionSelector
from codes.e_utils.logger import get_logger
from codes.e_utils.names import EnvironmentName, RLAlgorithmName

MODEL_SAVE_DIR = os.path.join(PROJECT_HOME, "out", "model_save_files")
if not os.path.exists(MODEL_SAVE_DIR):
    os.makedirs(MODEL_SAVE_DIR)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

os.environ["CUDA_VISIBLE_DEVICES"] = params.CUDA_VISIBLE_DEVICES_NUMBER_LIST

logger = get_logger("maplab_fedetated_play")


def play_main():
    env = rl_utils.get_environment(owner="worker", params=params)
    print("env:", params.ENVIRONMENT_ID)
    print("observation_space:", env.observation_space)
    print("action_space:", env.action_space)
    print("action_min: ", env.action_space.low[0], "action_max:", env.action_space.high[0])
    action_min = env.action_space.low[0]
    action_max = env.action_space.high[0]

    if params.ENVIRONMENT_ID in [EnvironmentName.PENDULUM_MATLAB_V0, EnvironmentName.PENDULUM_MATLAB_DOUBLE_RIP_V0]:
        env.start()

    agent, epsilon_tracker = rl_utils.get_rl_agent(
        env=env, worker_id=0, action_min=action_min, action_max=action_max, params=params
    )

    load_model(
        MODEL_SAVE_DIR, params.ENVIRONMENT_ID.value, agent.model
    )

    if params.RL_ALGORITHM == RLAlgorithmName.DQN_FAST_V0:
        action_selector = EpsilonGreedySomeTimesBlowDQNActionSelector(epsilon=0.0)
        agent.action_selector = action_selector
    elif params.RL_ALGORITHM == RLAlgorithmName.DDPG_FAST_V0:
        action_selector = EpsilonGreedySomeTimesBlowDDPGActionSelector(
            epsilon=0.0, ou_enabled=False, scale_factor=params.ACTION_SCALE
        )
        agent.action_selector = action_selector
    else:
        raise ValueError()

    num_step = 0
    num_episode = 0
    while True:
        done = False
        episode_reward = 0

        state = env.reset()

        num_episode += 1
        num_episode_step = 0
        while not done:
            num_step += 1
            num_episode_step += 1

            state = np.expand_dims(state, axis=0)
            if params.RL_ALGORITHM == RLAlgorithmName.DQN_FAST_V0:
                action, _, = agent(state)
            else:
                action, _, _ = agent(state)
            next_state, reward, done, info = env.step(action[0])
            state = next_state
            episode_reward += reward

            if num_step % 1000 == 0:
                print("EPISODE: {0}, EPISODE STEPS: {1}, TOTAL STEPS: {2}".format(
                    num_episode, num_episode_step, num_step
                ))

        print("EPISODE: {0}, EPISODE STEPS: {1}, TOTAL STEPS: {2}, EPISODE DONE --> EPISODE REWARD: {3}".format(
            num_episode, num_episode_step, num_step, episode_reward
        ))

        time.sleep(0.1)


if __name__ == "__main__":
    play_main()