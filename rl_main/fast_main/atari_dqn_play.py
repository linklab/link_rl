#!/usr/bin/env python3
import gym
import torch
import time
import os
import numpy as np
from common.common_utils import make_atari_env
from rl_main.fast_main.atari_dqn import MODEL_SAVE_DIR
from common.fast_rl import actions, value_based_model, rl_agent
from config.parameters import PARAMETERS as params

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
if torch.cuda.is_available():
    device = torch.device("cuda" if params.CUDA else "cpu")
else:
    device = torch.device("cpu")


def play_main():
    env = make_atari_env(params.ENVIRONMENT_ID.value, seed=0)

    net = value_based_model.DuelingDQNCNN(
        input_shape=env.observation_space.shape,
        n_actions=env.action_space.n
    ).to(device)
    print(net)

    rl_agent.load_model(MODEL_SAVE_DIR, params.ENVIRONMENT_ID.value, net.__name__, net, step=9431043)#1731249

    action_selector = actions.ArgmaxActionSelector()
    # action_selector = actions.EpsilonGreedyDQNActionSelector(epsilon=0.01)
    agent = rl_agent.DQNAgent(net, action_selector, device=device)

    done = False
    state = env.reset()

    episode_reward = 0
    while not done:
        # if episode_reward > 40:
        #     time.sleep(0.01)
        time.sleep(0.01)
        env.render()
        state = np.expand_dims(state, axis=0)
        action = agent(state)
        next_state, reward, done, info = env.step(action[0][0])

        # episode_reward += reward
        episode_reward += info['original_reward']
        if done and info['ale.lives'] != 0:
            done = False
            env.reset()

        state = next_state

    print(episode_reward)


if __name__ == "__main__":
    play_main()
