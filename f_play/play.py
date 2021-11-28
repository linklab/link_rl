import sys
import time

import gym
import torch
import os

CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))
PROJECT_HOME = os.path.abspath(os.path.join(CURRENT_PATH, os.pardir))
if PROJECT_HOME not in sys.path:
    sys.path.append(PROJECT_HOME)

from a_configuration.parameter import Parameter as params
from d_agents.off_policy.dqn.agent_dqn import AgentDqn
from g_utils.commons import AgentType, AgentMode, model_load


def play(env, agent, n_episodes):
    for i in range(n_episodes):
        episode_reward = 0  # cumulative_reward

        # Environment 초기화와 변수 초기화
        observation = env.reset()
        env.render()

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


def main_q_play(n_episodes):
    env = gym.make(params.ENV_NAME)
    n_features = env.observation_space.shape[0]
    n_actions = env.action_space.n

    if params.AGENT_TYPE == AgentType.Dqn:
        agent = AgentDqn(
            n_features=n_features,
            n_actions=n_actions,
            device=torch.device("cpu"),
            params=params
        )
    else:
        raise ValueError()

    model_load(
        model=agent.q_net,
        env_name=params.ENV_NAME,
        agent_type_name=params.AGENT_TYPE.name,
        file_name="500.0_0.0.pth"
    )
    play(env, agent, n_episodes=n_episodes)

    env.close()


if __name__ == "__main__":
    N_EPISODES = 5
    main_q_play(n_episodes=N_EPISODES)
