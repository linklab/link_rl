import sys
import time
import os
import warnings

import numpy as np

from a_configuration.a_base_config.a_environments.mujoco.gym_mujoco import ConfigMujoco
from a_configuration.a_base_config.a_environments.pybullet.gym_pybullet import ConfigBullet
from a_configuration.a_base_config.a_environments.unity.unity_box import ConfigUnityGymEnv
from a_configuration.a_base_config.c_models.recurrent_convolutional_models import ConfigRecurrentConvolutionalModel
from a_configuration.a_base_config.c_models.recurrent_linear_models import ConfigRecurrentLinearModel
from g_utils.commons import set_config
from g_utils.commons_rl import get_agent

warnings.filterwarnings("ignore")

import torch
from gym.spaces import Discrete, Box

from g_utils.types import AgentMode

CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))
PROJECT_HOME = os.path.abspath(os.path.join(CURRENT_PATH, os.pardir))
if PROJECT_HOME not in sys.path:
    sys.path.append(PROJECT_HOME)

from e_main.config_single import config
from g_utils.commons import model_load, get_single_env, get_env_info


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def play(env, agent, n_episodes):
    is_recurrent_model = any([
        isinstance(config.MODEL_TYPE, ConfigRecurrentLinearModel),
        isinstance(config.MODEL_TYPE, ConfigRecurrentConvolutionalModel)
    ])

    for i in range(n_episodes):
        episode_reward = 0  # cumulative_reward

        # Environment 초기화와 변수 초기화
        if isinstance(config, (ConfigMujoco, ConfigBullet)):
            env.render()
            observation = env.reset()
        else:
            observation = env.reset()
            env.render()

        if is_recurrent_model:
            observation = np.expand_dims(observation, axis=0)
            agent.model.init_recurrent_hidden()
            observation = [(observation, agent.model.recurrent_hidden)]

        episode_steps = 0

        while True:
            episode_steps += 1
            action = agent.get_action(observation, mode=AgentMode.PLAY)

            if isinstance(agent.action_space, Discrete):
                if action.ndim == 0:
                    scaled_action = action
                elif action.ndim == 1:
                    scaled_action = action[0]
                else:
                    raise ValueError()
            elif isinstance(agent.action_space, Box):
                if action.ndim == 1:
                    if agent.action_scale is not None:
                        scaled_action = action * agent.action_scale[0] + agent.action_bias[0]
                    else:
                        scaled_action = action
                elif action.ndim == 2:
                    if agent.action_scale is not None:
                        scaled_action = action[0] * agent.action_scale[0] + agent.action_bias[0]
                    else:
                        scaled_action = action[0]
                else:
                    raise ValueError()
            else:
                raise ValueError()

            # action을 통해서 next_state, reward, done, info를 받아온다
            next_observation, reward, done, _ = env.step(scaled_action)
            if is_recurrent_model:
                next_observation = np.expand_dims(next_observation, axis=0)
                next_observation = [(next_observation, agent.model.recurrent_hidden)]
            env.render()

            episode_reward += reward  # episode_reward 를 산출하는 방법은 감가률 고려하지 않는 이 라인이 더 올바름.
            observation = next_observation

            time.sleep(0.01)
            if done:
                break

        print("[EPISODE: {0}] EPISODE_STEPS: {1:3d}, EPISODE REWARD: {2:4.1f}".format(
            i, episode_steps, episode_reward
        ))


def main_play(n_episodes):
    set_config(config)

    observation_space, action_space = get_env_info(config)

    if isinstance(config, ConfigUnityGymEnv):
        config.NO_TEST_GRAPHICS = False

    env = get_single_env(config, config.NO_TEST_GRAPHICS)

    agent = get_agent(observation_space, action_space, config)

    model_load(
        model=agent.model,
        env_name=config.ENV_NAME,
        agent_type_name=config.AGENT_TYPE.name,
        file_name=config.PLAY_MODEL_FILE_NAME,
        config=config
    )
    play(env, agent, n_episodes=n_episodes)

    env.close()


if __name__ == "__main__":
    N_EPISODES = 5
    main_play(n_episodes=N_EPISODES)
