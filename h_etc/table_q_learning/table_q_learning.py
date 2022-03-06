# https://www.deeplearningwizard.com/deep_learning/deep_reinforcement_learning_pytorch/dynamic_programming_frozenlake/
# -*- coding: utf-8 -*-
import gym
import numpy as np
import matplotlib.pyplot as plt
import random

np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)


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


def greedy_action(action_values):
    max_value = np.max(action_values)
    return np.random.choice(
        [action_ for action_, value_ in enumerate(action_values) if value_ == max_value]
    )


def epsilon_greedy_action(action_values, epsilon):
    if np.random.rand() < epsilon:
        return random.choice(range(len(action_values)))
    else:
        max_value = np.max(action_values)
        return np.random.choice(
            [action_ for action_, value_ in enumerate(action_values) if value_ == max_value]
        )


def q_learning(num_episodes=500, num_test_episodes=7, alpha=0.1, gamma=0.95, epsilon=0.1):
    # Q-Table 초기화
    q_table = np.zeros([env.observation_space.n, env.action_space.n])

    episode_reward_list = []

    training_time_steps = 0
    last_episode_reward = 0

    for episode in range(num_episodes):
        episode_reward = 0  # cumulative_reward

        # Environment 초기화와 변수 초기화
        observation = env.reset()
        print("EPISODE: {0} - Initial State: {1}".format(episode, observation), end=" ")
        sList = [observation]

        episode_step = 0
        done = False

        # The Q-Table 알고리즘
        while not done:
            episode_step += 1
            action = epsilon_greedy_action(q_table[observation, :], epsilon)

            # action을 통해서 next_state, reward, done, info를 받아온다
            next_observation, reward, done, _ = env.step(action)

            episode_reward += reward

            # Q-Learning
            td_error = reward + gamma * np.max(q_table[next_observation, :]) - q_table[observation, action]
            q_table[observation, action] = q_table[observation, action] + alpha * td_error

            training_time_steps += 1  # Q-table 업데이트 횟수
            episode_reward_list.append(last_episode_reward)

            sList.append(next_observation)
            observation = next_observation

        print(sList, done, "GOAL" if done and observation == 15 else "")
        last_episode_reward = episode_reward

        if episode % 10 == 0:
            episode_reward_list_test, avg_episode_reward_test, std_episode_reward_test = q_learning_testing(
                num_test_episodes=num_test_episodes, q_table=q_table
            )
            print("[TEST RESULTS: {0} Episodes - {1}] Episode Reward Mean: {2:.3f}, Episode Reward Std.: {3:.3f}".format(
                3, episode_reward_list_test, avg_episode_reward_test, std_episode_reward_test
            ))
            if avg_episode_reward_test == 1.0 and std_episode_reward_test == 0.0:
                print("***** TRAINING DONE!!! *****")
                break

    return q_table, training_time_steps, episode_reward_list


def q_learning_testing(num_test_episodes, q_table):
    episode_reward_list = []

    for episode in range(num_test_episodes):
        episode_reward = 0  # cumulative_reward
        episode_step = 0

        observation = env.reset()

        done = False
        while not done:
            episode_step += 1
            action = greedy_action(q_table[observation, :])

            # action을 통해서 next_state, reward, done, info를 받아온다
            next_observation, reward, done, _ = env.step(action)

            episode_reward += reward  # episode_reward 를 산출하는 방법은 감가률 고려하지 않는 이 라인이 더 올바름.

            observation = next_observation

        episode_reward_list.append(episode_reward)

    return episode_reward_list, np.mean(episode_reward_list), np.std(episode_reward_list)


def main_q_table_learning():
    NUM_EPISODES = 200
    NUM_TEST_EPISODES = 7
    ALPHA = 0.1
    GAMMA = 0.95
    EPSILON = 0.1

    q_table, training_time_steps, episode_reward_list = q_learning(
        NUM_EPISODES, NUM_TEST_EPISODES, ALPHA, GAMMA, EPSILON
    )
    print("\nFinal Q-Table Values")
    print("    LEFT   DOWN  RIGHT     UP")
    for idx, observation in enumerate(q_table):
        print("{0:2d}".format(idx), end=":")
        for action_state in observation:
            print("{0:5.3f} ".format(action_state), end=" ")
        print()

    plt.plot(range(training_time_steps), episode_reward_list, color="Blue")
    plt.xlabel("training steps")
    plt.ylabel("episode reward")
    plt.show()


if __name__ == "__main__":
    main_q_table_learning()
