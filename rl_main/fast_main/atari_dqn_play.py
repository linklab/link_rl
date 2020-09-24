#!/usr/bin/env python3
import gym
import torch
import os
import numpy as np
from common.common_utils import make_atari_env
from rl_main.fast_main.atari_dqn import MODEL_SAVE_DIR
from common.fast_rl import actions, dqn_model, rl_agent
from config.parameters import PARAMETERS as params

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
device = torch.device("cuda" if params.CUDA else "cpu")


def play_main():
    env = make_atari_env(params)

    net = dqn_model.DQN(
        input_shape=env.observation_space.shape,
        n_actions=env.action_space.n
    ).to(device)
    print(net)

    dqn_model.load_model(MODEL_SAVE_DIR, params.ENVIRONMENT_ID.value, net.__name__, net, step=2950048)

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
