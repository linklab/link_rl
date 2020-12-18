#!/usr/bin/env python3
import time

import torch
import os

from common.logger import get_logger
from config.names import EnvironmentName, PROJECT_HOME, RLAlgorithmName
from rl_main import rl_utils

from common.fast_rl.rl_agent import float32_preprocessor
from common.fast_rl import actions, rl_agent
import numpy as np

from config.parameters import PARAMETERS as params

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

os.environ["CUDA_VISIBLE_DEVICES"] = params.CUDA_VISIBLE_DEVICES_NUMBER_LIST

logger = get_logger("maplab_fedetated_play")

MODEL_SAVE_DIR = os.path.join(PROJECT_HOME, "out", "model_save_files")


def play_main():
    env = rl_utils.get_environment(owner="worker", params=params)
    print("env:", params.ENVIRONMENT_ID)
    print("observation_space:", env.observation_space)
    print("action_space:", env.action_space)

    if params.ENVIRONMENT_ID in [EnvironmentName.PENDULUM_MATLAB_V0, EnvironmentName.PENDULUM_MATLAB_DOUBLE_AGENTS_V0]:
        env.start()

    rl_algorithm = rl_utils.get_rl_algorithm(env=env, worker_id=-1, logger=logger, params=params)

    rl_agent.load_model(
        MODEL_SAVE_DIR, params.ENVIRONMENT_ID.value, rl_algorithm.model.__name__, rl_algorithm.model
    )

    if params.RL_ALGORITHM == RLAlgorithmName.DQN_FAST_V0:
        action_selector = actions.EpsilonGreedyActionSelector(epsilon=params.EPSILON_INIT)

        agent = rl_agent.DQNAgent(rl_algorithm.model, action_selector, device=device)
    elif params.RL_ALGORITHM == RLAlgorithmName.DDPG_FAST_V0:
        action_min = -params.ACTION_SCALE
        action_max = params.ACTION_SCALE

        action_selector = actions.DDPGActionSelector(
            epsilon=0.0, ou_enabled=False, scale_factor=params.ACTION_SCALE
        )

        agent = rl_agent.AgentDDPG(
            rl_algorithm.model, n_actions=1, action_selector=action_selector,
            action_min=action_min, action_max=action_max, device=device, preprocessor=float32_preprocessor
        )
    else:
        raise ValueError()

    while True:
        done = False
        episode_reward = 0

        state = env.reset()
        state = np.expand_dims(state, axis=0)

        while not done:
            if params.RL_ALGORITHM == RLAlgorithmName.DQN_FAST_V0:
                action, _, = agent(state)
            else:
                action, _, _ = agent(state)
            next_state, reward, done, info = env.step(action[0])
            state = next_state
            state = np.expand_dims(state, axis=0)
            episode_reward += reward

        print(episode_reward)

        time.sleep(1)


if __name__ == "__main__":
    play_main()