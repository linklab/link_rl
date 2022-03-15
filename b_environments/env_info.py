# https://www.deeplearningwizard.com/deep_learning/deep_reinforcement_learning_pytorch/dynamic_programming_frozenlake/
import time

import gym
import random

# You can only register once
# To delete any new environment
# del gym.envs.registry.env_specs['FrozenLakeNotSlippery-v0']

# Make the environment based on deterministic policy
env = gym.make('CartPole-v1')


def env_info_details():
    #####################
    # observation space #
    #####################
    print("*" * 80)
    print(env.observation_space)

    for i in range(10):
        print(env.observation_space.sample())
    print()

    print("*" * 80)
    ################
    # action space #
    ################
    print(env.action_space)
    print(env.action_space.n)

    for i in range(10):
        print(env.action_space.sample(), end=" ")
    print()

    print("*" * 80)
    observation = env.reset()
    env.render()

    action = 1  # RIGHT
    next_observation, reward, done, info = env.step(action)
    env.render()
    print("Observation: {0}, Action: {1}, next_observation: {2}, Reward: {3}, Done: {4}, Info: {5}".format(
        observation, action, next_observation, reward, done, info
    ))

    observation = next_observation

    action = 0  # DOWN
    next_observation, reward, done, info = env.step(action)
    env.render()

    print("Observation: {0}, Action: {1}, next_observation: {2}, Reward: {3}, Done: {4}, Info: {5}".format(
        observation, action, next_observation, reward, done, info
    ))

    print("*" * 80)
    print("*" * 80)
    print("*" * 80)


class Dummy_Agent:
    def get_action(self, observation):
        available_action_ids = [0, 1]
        action_id = random.choice(available_action_ids)
        return action_id


def run_env():
    print("START RUN!!!")
    agent = Dummy_Agent()

    for i in range(100):
        observation = env.reset()
        env.render()
        done = False

        while not done:
            time.sleep(0.05)
            action = agent.get_action(observation)
            next_observation, reward, done, info = env.step(action)
            #env.render()
            print("Observation: {0}, Action: {1}, next_observation: {2}, Reward: {3}, Done: {4}, Info: {5}".format(
                observation, action, next_observation, reward, done, info
            ))
            observation = next_observation

        time.sleep(1)


if __name__ == "__main__":
    env_info_details()
    run_env()
