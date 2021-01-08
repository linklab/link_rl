# https://github.com/openai/gym/blob/master/gym/envs/classic_control/pendulum.py
# https://mspries.github.io/jimmy_pendulum.html
#!/usr/bin/env python3
import os, sys

from rl_main import rl_utils

current_path = os.path.dirname(os.path.realpath(__file__))
PROJECT_HOME = os.path.abspath(os.path.join(current_path, os.pardir, os.pardir))

if PROJECT_HOME not in sys.path:
    sys.path.append(PROJECT_HOME)

from codes.a_config.parameters import PARAMETERS as params
import numpy as np

np.set_printoptions(formatter={'float_kind': lambda x: '{0:0.6f}'.format(x)})

ACTION_SCALE_FACTOR = 0.035

env = rl_utils.get_environment(owner="worker", params=params)
print("env:", params.ENVIRONMENT_ID)
print("observation_space:", env.observation_space)
print("action_space:", env.action_space)


MAX_GLOBAL_STEP = 50000


def main():
    done_episode = 0
    step = 0

    while step < MAX_GLOBAL_STEP:
        episode_reward = 0.0
        done = False
        state = env.reset()
        episode_step = 0
        action_left = 0
        action_right = 20

        action = action_left

        while not done:
            if action == action_left and episode_step % 10 == 0:
                action = action_right
            elif action == action_right and episode_step % 10 == 0:
                action = action_left
            else:
                pass

            next_state, reward, done, info = env.step(action=action)
            print("[Step:{0}] {1:2d}, {2:6.3f}, {3:6.3f}, {4:6.3f}, {5}".format(
                step, action, info["adjusted_pendulum_1_radian"], info["adjusted_pendulum_2_radian"], reward, done
            ), flush=True)
            episode_step += 1
            state = next_state
            episode_reward += reward
            step += 1

        done_episode += 1
        print("[{0:5d}/{1}] Episode: {2:3d}, Episode Reward: {3:6.1f}".format(
            step, MAX_GLOBAL_STEP, done_episode, episode_reward
        ), flush=True)


if __name__ == "__main__":
    main()