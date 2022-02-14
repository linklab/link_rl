# https://www.deeplearningwizard.com/deep_learning/deep_reinforcement_learning_pytorch/dynamic_programming_frozenlake/
import gym
import random

# Environment
# Slippery environment (stochastic policy, move left probability = 1/3) comes by default!
# If we want deterministic policy, we need to create new environment
# Make environment No Slippery (deterministic policy, move left = 100% left)
MAX_EPISODE_STEPS = 250

gym.envs.register(
    id='FrozenLakeNotSlippery-v0',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={
        'map_name': '4x4', 'is_slippery': False
    },
    max_episode_steps=MAX_EPISODE_STEPS,
)

# You can only register once
# To delete any new environment
# del gym.envs.registry.env_specs['FrozenLakeNotSlippery-v0']

# Make the environment based on deterministic policy
env = gym.make('FrozenLakeNotSlippery-v0')
# env = gym.make('FrozenLake-v0')


def env_info_details():
    #####################
    # observation space #
    #####################
    print("*" * 80)
    print(env.observation_space)
    print(env.observation_space.n)
    # We should expect to see 15 possible grids from 0 to 15 when
    # we uniformly randomly sample from our observation space
    for i in range(10):
        print(env.observation_space.sample(), end=" ")
    print()

    print("*" * 80)
    ################
    # action space #
    ################
    print(env.action_space)
    print(env.action_space.n)
    # We should expect to see 4 actions when
    # we uniformly randomly sample:
    #     1. LEFT: 0
    #     2. DOWN: 1
    #     3. RIGHT: 2
    #     4. UP: 3
    for i in range(10):
        print(env.action_space.sample(), end=" ")
    print()

    print("*" * 80)
    # This sets the initial state at S, our starting point
    # We can render the environment to see where we are on the 4x4 frozenlake gridworld
    observation = env.reset()
    env.render()

    action = 2  # RIGHT
    next_observation, reward, done, info = env.step(action)
    env.render()

    # Prob = 1: deterministic policy, if we choose to go right, we'll go right
    print("Observation: {0}, Action: {1}, next_observation: {2}, Reward: {3}, Done: {4}, Info: {5}".format(
        observation, action, next_observation, reward, done, info
    ))

    observation = next_observation

    action = 1  # DOWN
    next_observation, reward, done, info = env.step(action)
    env.render()

    print("Observation: {0}, Action: {1}, next_observation: {2}, Reward: {3}, Done: {4}, Info: {5}".format(
        observation, action, next_observation, reward, done, info
    ))

    print("*" * 80)
    print("*" * 80)
    print("*" * 80)


class DummyAgent:
    def get_action(self, observation):
        available_action_ids = [0, 1, 2, 3]
        action_id = random.choice(available_action_ids)
        return action_id


def run_env():
    print("START RUN!!!")
    agent = DummyAgent()
    observation = env.reset()
    env.render()

    done = False

    while not done:
        action = agent.get_action(observation)
        next_observation, reward, done, info = env.step(action)
        env.render()
        print("Observation: {0}, Action: {1}, next_observation: {2}, Reward: {3}, Done: {4}, Info: {5}".format(
            observation, action, next_observation, reward, done, info
        ))
        observation = next_observation


if __name__ == "__main__":
    env_info_details()
    run_env()
