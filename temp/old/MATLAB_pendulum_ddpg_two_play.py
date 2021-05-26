#!/usr/bin/env python3
import torch
import os

from config.names import PROJECT_HOME

MODEL_SAVE_DIR = os.path.join(PROJECT_HOME, "out", "model_save_files")

print(torch.__version__)

from common.fast_rl.rl_agent import float32_preprocessor
from common.fast_rl import actions, rl_agent, policy_based_model
import numpy as np

from config.parameters import PARAMETERS as params

from common.environments import MatlabRotaryInvertedPendulumEnv
import math
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
if torch.cuda.is_available():
    device = torch.device("cuda" if params.CUDA else "cpu")
else:
    device = torch.device("cpu")

if params.CH:
    SWING_UP_SCALE_FACTOR = 0.035
    BALANCING_SCALE_FACTOR = 0.001
else:
    SWING_UP_SCALE_FACTOR = 0.035
    BALANCING_SCALE_FACTOR = 0.001


def play_main():
    env = MatlabRotaryInvertedPendulumEnv()
    print("env:", params.ENVIRONMENT_ID)
    print("observation_space:", 3)
    print("action_space:", 1)

    env.start()

    swing_up_action_min = -SWING_UP_SCALE_FACTOR
    swing_up_action_max = SWING_UP_SCALE_FACTOR
    balancing_action_min = -BALANCING_SCALE_FACTOR
    balancing_action_max = BALANCING_SCALE_FACTOR
    count_bal = 0

    actor_net = policy_based_model.DDPGActor(
        obs_size=4,
        hidden_size_1=512, hidden_size_2=256,
        n_actions=1,
        scale=SWING_UP_SCALE_FACTOR
    ).to(device)
    print(actor_net)

    rl_agent.load_model(MODEL_SAVE_DIR, params.ENVIRONMENT_ID, actor_net.__name__, actor_net)
    # action_selector = actions.EpsilonGreedyDDPGActionSelector(epsilon=params.EPSILON_INIT)

    action_selector = actions.EpsilonGreedyDDPGActionSelector(epsilon=0.0, ou_enabled=False, scale_factor=SWING_UP_SCALE_FACTOR)

    agent = rl_agent.AgentDDPG(
        actor_net, n_actions=1, action_selector=action_selector,
        action_min=swing_up_action_min, action_max=swing_up_action_max, device=device, ou_enabled=False,
        preprocessor=float32_preprocessor
    )

    actor_balance_net = policy_based_model.DDPGActor(
        obs_size=4,
        hidden_size_1=512, hidden_size_2=256,
        n_actions=1,
        scale=BALANCING_SCALE_FACTOR
    ).to(device)
    print(actor_balance_net)

    rl_agent.load_model(MODEL_SAVE_DIR, params.ENVIRONMENT_ID, actor_balance_net.__name__, actor_balance_net)
    # action_selector = actions.EpsilonGreedyDDPGActionSelector(epsilon=params.EPSILON_INIT)

    action_selector_balance = actions.EpsilonGreedyDDPGActionSelector(epsilon=0.0, ou_enabled=False, scale_factor=BALANCING_SCALE_FACTOR)

    agent_balance = rl_agent.AgentDDPG(
        actor_balance_net, n_actions=1, action_selector=action_selector_balance,
        action_min=balancing_action_min, action_max=balancing_action_max, device=device, ou_enabled=False,
        preprocessor=float32_preprocessor
    )


    done = False
    state = env.reset()

    step = 0
    state = np.expand_dims(state, axis=0)
    while not done:
        if math.cos(math.pi) < state[0][0] < math.cos(3.316125):  # cos(180) < exp[0][0] < cos(190) (-1<exp[0][0]<-0.98480)
            count_bal += 1
        else:
            count_bal = 0

        if count_bal < 10:
            action = agent(state)
            next_state, reward, done, info = env.step(action[0][-1])
            print(step, state, action[0][-1], next_state, reward, done)
            state = next_state
            state = np.expand_dims(state, axis=0)
            step += 1
        else:
            action = agent_balance(state)
            next_state, reward, done, info = env.step(action[0][-1])
            print(step, state, action[0][-1], next_state, reward, done)
            state = next_state
            state = np.expand_dims(state, axis=0)
            step += 1



if __name__ == "__main__":
    play_main()