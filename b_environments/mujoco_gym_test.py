import sys
import time
import os

import gym


def play(env, n_episodes):
    for i in range(n_episodes):
        episode_reward = 0  # cumulative_reward

        env.render()
        observation = env.reset()

        episode_steps = 0

        while True:
            episode_steps += 1
            action = env.action_space.sample()

            next_observation, reward, done, _ = env.step(action)
            env.render()

            episode_reward += reward  # episode_reward 를 산출하는 방법은 감가률 고려하지 않는 이 라인이 더 올바름.
            observation = next_observation

            time.sleep(0.05)
            if done:
                break

        print("[EPISODE: {0}] EPISODE_STEPS: {1:3d}, EPISODE REWARD: {2:4.1f}".format(
            i, episode_steps, episode_reward
        ))

    env.close()


def main():
    # env = gym.make("Ant-v2"); play(env, n_episodes=5)
    # env = gym.make("Walker2d-v2"); play(env, n_episodes=5)
    # env = gym.make("Hopper-v2");play(env, n_episodes=5)
    # env = gym.make("HalfCheetah-v2");play(env, n_episodes=1)

    # env = gym.make("InvertedDoublePendulum-v2");play(env, n_episodes=10)
    # env = gym.make("Reacher-v2");play(env, n_episodes=3)
    # env = gym.make("Pusher-v2");play(env, n_episodes=3)
    # env = gym.make("Thrower-v2");play(env, n_episodes=5)
    # env = gym.make("Striker-v2");play(env, n_episodes=5)
    # env = gym.make("Swimmer-v2");play(env, n_episodes=5)
    # env = gym.make("Humanoid-v2");play(env, n_episodes=5)
    env = gym.make("HumanoidStandup-v2");play(env, n_episodes=5)


if __name__ == "__main__":
    main()
