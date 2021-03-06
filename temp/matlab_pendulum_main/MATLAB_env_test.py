# https://github.com/openai/gym/blob/master/gym/envs/classic_control/pendulum.py
# https://mspries.github.io/jimmy_pendulum.html
#!/usr/bin/env python3
import os, sys

current_path = os.path.dirname(os.path.realpath(__file__))
PROJECT_HOME = os.path.abspath(os.path.join(current_path, os.pardir, os.pardir))

if PROJECT_HOME not in sys.path:
    sys.path.append(PROJECT_HOME)

from common.environments import MatlabRotaryInvertedPendulumEnv
from codes.a_config.parameters import PARAMETERS as params
import random

ACTION_SCALE_FACTOR = 0.035

env = MatlabRotaryInvertedPendulumEnv(
    action_min=ACTION_SCALE_FACTOR * -1.0, action_max=ACTION_SCALE_FACTOR
)
print("env:", params.ENVIRONMENT_ID)
print("observation_space:", 4)
print("action_space:", 1)

MAX_GLOBAL_STEP = 10000


def main():
    done_episode = 0
    step = 0

    while step < MAX_GLOBAL_STEP:
        episode_reward = 0.0
        done = False
        env.reset()
        while not done:
            action = random.uniform(ACTION_SCALE_FACTOR * -1.0, ACTION_SCALE_FACTOR)
            next_state, reward, done, _ = env.step(action=action)
            episode_reward += reward
            step += 1

        done_episode += 1
        print("[{0:5d}/{1}] Episode: {2:3d}, Episode Reward: {3:6.1f}".format(
            step, MAX_GLOBAL_STEP, done_episode, episode_reward
        ), flush=True)


if __name__ == "__main__":
    main()