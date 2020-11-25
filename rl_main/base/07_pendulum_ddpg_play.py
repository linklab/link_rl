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

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
if torch.cuda.is_available():
    device = torch.device("cuda" if params.CUDA else "cpu")
else:
    device = torch.device("cpu")


def play_main():
    env = gym.make(params.ENVIRONMENT_ID.value)

    print(env.action_space.low[0], env.action_space.high[0])
    action_min = env.action_space.low[0]
    action_max = env.action_space.high[0]

    actor_net = policy_based_model.DDPGActor(
        obs_size=3,
        hidden_size_1=512, hidden_size_2=256,
        n_actions=1,
        scale=1.0
    ).to(device)
    print(actor_net)

    rl_agent.load_model(MODEL_SAVE_DIR, params.ENVIRONMENT_ID.value, actor_net.__name__, actor_net)

    action_selector = actions.DDPGActionSelector(epsilon=0.0, ou_enabled=False, scale_factor=1.0)

    agent = rl_agent.AgentDDPG(
        actor_net, n_actions=1, action_selector=action_selector,
        action_min=action_min, action_max=action_max, device=device, preprocessor=float32_preprocessor
    )

    done = False
    state = env.reset()

    step = 0
    while not done:
        env.render()
        state = np.expand_dims(state, axis=0)
        action = agent(state)
        next_state, reward, done, info = env.step(action[0][0])
        state = next_state
        print(step)
        step += 1


if __name__ == "__main__":
    play_main()