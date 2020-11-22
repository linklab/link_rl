# https://github.com/openai/gym/blob/master/gym/envs/classic_control/pendulum.py
# https://mspries.github.io/jimmy_pendulum.html
#!/usr/bin/env python3
import torch
import os, sys

idx = os.getcwd().index("link_rl")
PROJECT_HOME = os.getcwd()[:idx] + "link_rl"
sys.path.append(PROJECT_HOME)

from common.environments.matlab.matlabenv import MatlabRotaryInvertedPendulumEnv

from config.parameters import PARAMETERS as params

def main():
    env = MatlabRotaryInvertedPendulumEnv()
    print("env:", params.ENVIRONMENT_ID)
    print("observation_space:", 4)
    print("action_space:", 1)

    env.start()

    env.reset()
    done = False

    action = -1.0
    while not done:
        next_state, reward, done, _ = env.step(action=action)
        #action = -action


if __name__ == "__main__":
    main()