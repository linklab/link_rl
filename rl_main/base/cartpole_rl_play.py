#!/usr/bin/env python3
import gym
import torch
import os

from rl_main.base.cartpole_rl_main import cuda, MODEL_SAVE_DIR, env_name

print(torch.__version__)

from common.fast_rl import actions, dqn_model, rl_agent
import numpy as np

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
device = torch.device("cuda" if cuda else "cpu")

"""
Utility functions to enable video recording of gym environment and displaying it
To enable video, just do "env = wrap_env(env)""
"""


def play_main():
    env = gym.make(env_name)

    net = dqn_model.DuelingDQNMLP(
        obs_size=4,
        hidden_size_1=128, hidden_size_2=128,
        n_actions=2
    ).to(device)
    print(net)

    dqn_model.load_model(MODEL_SAVE_DIR, env_name, net.__name__, net, step=14780)

    action_selector = actions.ArgmaxActionSelector()
    agent = rl_agent.DQNAgent(net, action_selector, device=device)

    done = False
    state = env.reset()

    while not done:
        env.render()
        state = np.expand_dims(state, axis=0)
        action = agent(state)
        next_state, reward, done, info = env.step(action[0][0])
        state = next_state


if __name__ == "__main__":
    play_main()