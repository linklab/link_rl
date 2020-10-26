#!/usr/bin/env python3
import gym
import torch
import os

MODEL_SAVE_DIR = os.path.join(".", "saved_models")

print(torch.__version__)

from common.fast_rl.rl_agent import float32_preprocessor
from common.fast_rl import actions, value_based_model, rl_agent, policy_based_model
import numpy as np

from config.parameters import PARAMETERS as params

from common.environments.matlab.matlabenv import MatlabRotaryInvertedPendulumEnv

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
if torch.cuda.is_available():
    device = torch.device("cuda" if params.CUDA else "cpu")
else:
    device = torch.device("cpu")

if params.CH:
    SCALE_FACTOR = 0.01
else:
    SCALE_FACTOR = 0.025

def play_main():
    env = MatlabRotaryInvertedPendulumEnv()
    env.start()

    action_min = -SCALE_FACTOR
    action_max = SCALE_FACTOR

    actor_net = policy_based_model.DDPGActor(
        obs_size=4,
        hidden_size_1=512, hidden_size_2=256,
        n_actions=1,
        scale=SCALE_FACTOR
    ).to(device)
    print(actor_net)

    rl_agent.load_model(MODEL_SAVE_DIR, params.ENVIRONMENT_ID, actor_net.__name__, actor_net)
    # action_selector = actions.EpsilonGreedyDDPGActionSelector(epsilon=params.EPSILON_INIT)

    action_selector = actions.DDPGActionSelector(epsilon=0.0, ou_enabled=False)

    agent = rl_agent.AgentDDPG(
        actor_net, n_actions=1, action_selector=action_selector, action_min=action_min, action_max=action_max, device=device, ou_enabled=False,
        preprocessor=float32_preprocessor
    )

    done = False
    state = env.reset()

    step = 0
    while not done:
        state = np.expand_dims(state, axis=0)
        action = agent(state)
        print(action)
        next_state, reward, done, info = env.step(action[-1])
        state = next_state
        print(step)
        step += 1


if __name__ == "__main__":
    play_main()