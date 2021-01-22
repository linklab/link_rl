from gym import RewardWrapper


class RewardChanger(RewardWrapper):
    r"""Change the reward via an arbitrary function.

    Example::

        >>> import gym
        >>> env = gym.make('CartPole-v1')
        >>> env = TransformReward(env, lambda r: 0.01*r)
        >>> env.reset()
        >>> observation, reward, done, info = env.step(env.action_space.sample())
        >>> reward
        0.01

    Args:
        env (Env): environment
        f (callable): a function that transforms the reward

    """
    def __init__(self, env, f, reverse_f):
        super(RewardChanger, self).__init__(env)
        assert callable(f)
        assert callable(reverse_f)
        self.f = f
        self.reverse_f = reverse_f

    def reward(self, reward):
        return self.f(reward)

    def reverse_reward(self, changed_reward):
        return self.reverse_f(changed_reward)
