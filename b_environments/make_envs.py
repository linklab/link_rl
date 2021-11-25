import gym


def make_gym_env(env_name):
    def _make():
        env = gym.make(env_name)
        return env

    return _make
