#!/usr/bin/env python3
import gym
import torch
import os

MODEL_SAVE_DIR = os.path.join(".", "saved_models")

print(torch.__version__)

from common.fast_rl import actions, value_based_model, rl_agent
import numpy as np

from config.parameters import PARAMETERS as params

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
if torch.cuda.is_available():
    device = torch.device("cuda" if params.CUDA else "cpu")
else:
    device = torch.device("cpu")


def play_main():
    env = gym.make(params.ENVIRONMENT_ID.value)

    net = value_based_model.DuelingDQNMLP(
        obs_size=4,
        hidden_size_1=128, hidden_size_2=128,
        n_actions=2
    ).to(device)
    print(net)

    rl_agent.load_model(MODEL_SAVE_DIR, params.ENVIRONMENT_ID.value, net.__name__, net)

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