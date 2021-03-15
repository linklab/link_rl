import collections
import numpy as np
import gym
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


def counts_hash(obs):
    if type(obs) not in (tuple, list):
        obs = obs.tolist()

    # round(v, 1): (0.2, 1.0, -19.3, 1.0, -0.2, -27.1, 0.5, 0.9, 12.6)
    # round(v, 0): (0, 1, -19, 1, 0, -27, 1, 1, 12)
    hashed_obs = tuple(map(lambda v: round(v, 0), obs))

    # print(hashed_obs)

    return hashed_obs


class PseudoCountRewardWrapper(gym.Wrapper):
    def __init__(self, env, hash_function=counts_hash, count_based_reward_scale=0.2):
        super(PseudoCountRewardWrapper, self).__init__(env)
        self.hash_function = hash_function
        self.count_based_reward_scale = count_based_reward_scale
        self.counts = collections.Counter()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        observation_uncertainty = self.get_observation_uncertainty(obs)
        intrinsic_reward = reward * self.count_based_reward_scale * observation_uncertainty
        info["observation_uncertainty"] = observation_uncertainty
        return obs, reward + intrinsic_reward, done, info

    def get_observation_uncertainty(self, obs) -> float:
        """
        Increments observation counter and returns pseudo-count reward
        :param obs: observation
        :return: extra reward
        """
        h = self.hash_function(obs)
        self.counts[h] += 1
        return np.sqrt(1.0 / self.counts[h])
