import gym

class VectorEnv:
    def __init__(self, make_env_fn, n):
        self.envs = tuple(make_env_fn() for _ in range(n))

    # Call this only once at the beginning of training (optional):
    def seed(self, seeds):
        assert len(self.envs) == len(seeds)
        return tuple(env.seed(s) for env, s in zip(self.envs, seeds))

    # Call this only once at the beginning of training:
    def reset(self):
        return tuple(env.reset() for env in self.envs)

    # Call this on every timestep:
    def step(self, actions):
        assert len(self.envs) == len(actions)
        return_values = []
        for env, a in zip(self.envs, actions):
            observation, reward, done, info = env.step(a)
            if done:
                observation = env.reset()
            return_values.append((observation, reward, done, info))
        return tuple(return_values)

    # Call this at the end of training:
    def close(self):
        for env in self.envs:
            env.close()

make_env_fn = lambda: gym.make('CartPole-v0')
env = VectorEnv(make_env_fn, n=4)