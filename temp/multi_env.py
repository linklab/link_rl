import gym

from gym.vector import SyncVectorEnv, AsyncVectorEnv
from gym.vector.tests.utils import make_env


def test_create_sync_vector_env():
    env_fns = [make_env('CartPole-v0', i) for i in range(8)]
    env = AsyncVectorEnv(env_fns)
    assert env.num_envs == 8

    observation_space = env.observation_space
    print(observation_space)

    action_space = env.action_space
    print(action_space)

    while True:
        state = env.reset()
        done = False
        while not done:
            action = action_space.sample()
            next_state, reward, done, info = env.step(action)
            print(state, action, next_state, reward, done, info)
            state = next_state


if __name__ == "__main__":
    test_create_sync_vector_env()
