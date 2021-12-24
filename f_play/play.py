import sys
import time

import torch
import os

from e_main.supports.main_preamble import get_agent
from g_utils.types import AgentMode

CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))
PROJECT_HOME = os.path.abspath(os.path.join(CURRENT_PATH, os.pardir))
if PROJECT_HOME not in sys.path:
    sys.path.append(PROJECT_HOME)

from e_main.parameter import Parameter
from g_utils.commons import model_load, get_test_env


parameter = Parameter()


def play(env, agent, n_episodes):
    for i in range(n_episodes):
        episode_reward = 0  # cumulative_reward

        # Environment 초기화와 변수 초기화
        env.render()
        observation = env.reset()

        episode_steps = 0

        while True:
            episode_steps += 1
            action = agent.get_action(observation, mode=AgentMode.PLAY)

            # action을 통해서 next_state, reward, done, info를 받아온다
            next_observation, reward, done, _ = env.step(action)
            env.render()

            episode_reward += reward  # episode_reward 를 산출하는 방법은 감가률 고려하지 않는 이 라인이 더 올바름.
            observation = next_observation

            time.sleep(0.05)
            if done:
                break

        print("[EPISODE: {0}] EPISODE_STEPS: {1:3d}, EPISODE REWARD: {2:4.1f}".format(
            i, episode_steps, episode_reward
        ))


def main_play(n_episodes):
    env, observation_shape, n_actions = get_test_env(parameter)

    agent = get_agent(
        observation_shape, n_actions, device=torch.device("cpu"), parameter=parameter,
        max_training_steps=parameter.MAX_TRAINING_STEPS
    )

    model_load(
        model=agent.q_net,
        env_name=parameter.ENV_NAME,
        agent_type_name=parameter.AGENT_TYPE.name,
        file_name="200.0_0.0_2021_12_18.pth",
        parameter=parameter
    )
    play(env, agent, n_episodes=n_episodes)

    env.close()


if __name__ == "__main__":
    N_EPISODES = 5
    main_play(n_episodes=N_EPISODES)
