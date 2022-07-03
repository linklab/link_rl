# -*- coding: utf-8 -*-
import os
import sys
import time

import gym

from link_rl.g_utils.commons import get_train_env, get_single_env

CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))
PROJECT_HOME = os.path.abspath(os.path.join(CURRENT_PATH, os.pardir))
if PROJECT_HOME not in sys.path:
    sys.path.append(PROJECT_HOME)

#ENV_NAME = "PongNoFrameskip-v4"
ENV_NAME = "ALE/Pong-v5"


def main_env_info():
    env = gym.make(ENV_NAME, render_mode="human", frameskip=4, repeat_action_probability=0.0)
    env = gym.wrappers.AtariPreprocessing(env, frame_skip=1, grayscale_obs=True, scale_obs=True)
    env = gym.wrappers.FrameStack(env, num_stack=4, lz4_compress=True)

    print("*" * 80)
    print(env.observation_space)
    # for i in range(1):
    #     print(env.observation_space.sample())
    # print()

    ################
    # action space #
    ################
    print("*" * 80)
    print(env.action_space)
    print(env.action_space.n)
    print(env.get_action_meanings())
    for i in range(10):
        print(env.action_space.sample(), end=" ")
    print()

    print("*" * 80)
    # Starting State:
    # All observations are assigned a uniform random value in [-0.05..0.05]
    observation = env.reset()
    print(observation.shape)

    # float32
    # 32 * 4 * 84 * 84 = 110KB
    # 110KB * 1,000,000 = 110,250,000KB = 105GB

    # Reward:
    # Reward is 1 for every step taken, including the termination step
    action = 0  # LEFT
    next_observation, reward, done, info = env.step(action)

    # Observation = 1: move to grid number 1 (unchanged)
    # Prob = 1: deterministic policy, if we choose to go right, we'll go right
    print("Observation: {0}, Action: {1}, Next Observation: {3}, Reward: {2}, Done: {4}, Info: {5}".format(
        observation.shape, action, next_observation.shape, reward, done, info
    ))

    observation = next_observation

    action = 1
    next_observation, reward, done, info = env.step(action)

    print("Observation: {0}, Action: {1}, Next Observation: {4}, Reward: {3}, Done: {4}, Info: {5}".format(
        observation.shape, action, next_observation.shape, reward, done, info
    ))

    print("*" * 80)
    # This sets the initial state at S, our starting point
    # We can render the environment to see where we are on the 4x4 frozenlake gridworld
    observation = env.reset()
    #env.render()

    # actions = [0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5]

    actions = ([2] * 3 + [3] * 3) * 500

    for action in actions:
        time.sleep(0.05)
        next_observation, reward, done, info = env.step(action)
        #env.render()
        print("Observation: {0}, Action: {1}, Next Observation: {2}, Reward: {3}, Done: {4}, Info: {5}".format(
            observation.shape, action, next_observation.shape, reward, done, info
        ))
        observation = next_observation

    env.close()

def main_env_info2(config):
    class Dummy_Agent:
        def __init__(self):
            pass

        def get_action(self, observation):
            import numpy as np
            action = np.random.randint(0, 2)
            return action

    env = get_single_env(config)
    agent = Dummy_Agent()

    total_episodes = 10_000
    total_time_steps = 1

    for ep in range(1, total_episodes + 1):
        observation, info = env.reset(return_info=True)
        done = False
        episode_time_steps = 1

        while not done:
            action = agent.get_action(observation)
            next_observation, reward, done, info = env.step(action)

            print("[Ep.: {0:4}, Time Steps: {1:4}/{2:4}] "
                  "Obs.: {3}, Action: {4}, Next_obs.: {5}, Reward: {6}, Done: {7}, Info: {8}".format(
                ep, episode_time_steps, total_time_steps,
                str(observation.shape), str(action), str(next_observation.shape), str(reward), done, info
            ))
            observation = next_observation
            total_time_steps += 1
            episode_time_steps += 1


if __name__ == "__main__":
    #main_env_info()

    from link_rl.a_configuration.b_single_config.open_ai_gym.atari.config_pong import ConfigPongDqn
    config = ConfigPongDqn()
    config.MODEL_CREATOR_TYPE = "QModelCreatorGymAtariConv"

    main_env_info2(config)
