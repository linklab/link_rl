import sys
import time
import os
import warnings

from a_configuration.b_base.a_environments.pybullet.gym_mujoco import ParameterMujoco
from a_configuration.b_base.a_environments.pybullet.gym_pybullet import ParameterBullet

warnings.filterwarnings("ignore")

import torch
from gym.spaces import Discrete, Box

from e_main.supports.main_preamble import get_agent
from g_utils.types import AgentMode

CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))
PROJECT_HOME = os.path.abspath(os.path.join(CURRENT_PATH, os.pardir))
if PROJECT_HOME not in sys.path:
    sys.path.append(PROJECT_HOME)

from e_main.parameter import parameter
from g_utils.commons import model_load, get_single_env, get_env_info


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def play(env, agent, n_episodes):
    for i in range(n_episodes):
        episode_reward = 0  # cumulative_reward

        # Environment 초기화와 변수 초기화
        if isinstance(parameter, (ParameterMujoco, ParameterBullet)):
            env.render()
            observation = env.reset()
        else:
            observation = env.reset()
            env.render()

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
                    if agent.action_scale_factor is not None:
                        scaled_action = action * agent.action_scale_factor[0]
                    else:
                        scaled_action = action
                elif action.ndim == 2:
                    if agent.action_scale_factor is not None:
                        scaled_action = action[0] * agent.action_scale_factor[0]
                    else:
                        scaled_action = action[0]
                else:
                    raise ValueError()
            else:
                raise ValueError()

            # action을 통해서 next_state, reward, done, info를 받아온다
            next_observation, reward, done, _ = env.step(scaled_action)
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
    observation_space, action_space = get_env_info(parameter)
    env = get_single_env(parameter)

    agent = get_agent(observation_space, action_space, parameter)

    model_load(
        model=agent.model,
        env_name=parameter.ENV_NAME,
        agent_type_name=parameter.AGENT_TYPE.name,
        file_name=parameter.PLAY_MODEL_FILE_NAME,
        parameter=parameter
    )
    play(env, agent, n_episodes=n_episodes)

    env.close()


if __name__ == "__main__":
    N_EPISODES = 5
    main_play(n_episodes=N_EPISODES)
