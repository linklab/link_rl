import collections
import numpy as np
import gym
from gym import RewardWrapper
import pickle

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


def counts_hash(obs, precision, filter):
    if type(obs) not in (tuple, list):
        obs = obs.tolist()

    # round(v, 1): (0.2, 1.0, -19.3, 1.0, -0.2, -27.1, 0.5, 0.9, 12.6)
    # round(v, 0): (0, 1, -19, 1, 0, -27, 1, 1, 12)

    hashed_obs = np.asarray(tuple(map(lambda v: round(v, precision), obs)))
    hashed_obs = tuple(np.extract(filter, hashed_obs))
    # print(hashed_obs[:6])

    return hashed_obs


class PseudoCountRewardWrapper(gym.Wrapper):
    def __init__(self, env, hash_function=counts_hash, params=None):
        super(PseudoCountRewardWrapper, self).__init__(env)
        self.hash_function = hash_function
        self.params = params
        self.count_based_reward_scale = params.COUNT_BASED_REWARD_SCALE
        self.precision = params.COUNT_BASED_PRECISION

        self.counts = collections.Counter()
        self.global_uncertainty = 1.0
        self.global_uncertainty_list = []
        self.step_idx = 0

        if self.params.COUNT_BASED_FILTER:
            self.filter = self.params.COUNT_BASED_FILTER
        else:
            pass

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        observation_uncertainty = self.get_observation_uncertainty(obs)
        intrinsic_reward = self.count_based_reward_scale * observation_uncertainty
        info["intrinsic_reward"] = intrinsic_reward

        # self.global_uncertainty = ((self.global_uncertainty * self.step_idx) + observation_uncertainty) / (self.step_idx + 1.0)
        self.global_uncertainty = np.sqrt(len(self.counts) / sum(self.counts.values()))

        # print(len(self.counts), sum(self.counts.values()), self.global_uncertainty)

        info["global_uncertainty"] = self.global_uncertainty
        self.step_idx += 1

        # if self.step_idx % 10000 == 0:
        #     print("*global_uncertainty = {0:0.5}".format(self.global_uncertainty))
        #     self.global_uncertainty_list.append(self.global_uncertainty)
        #     with open('global_uncertainty_list.pickle', 'wb') as f:
        #         pickle.dump(self.global_uncertainty_list, f)

        return obs, reward + intrinsic_reward, done, info

    def get_observation_uncertainty(self, obs) -> float:
        """
        Increments observation counter and returns pseudo-count reward
        :param obs: observation
        :return: extra reward
        """
        h = self.hash_function(obs, self.precision, self.params.COUNT_BASED_FILTER)
        self.counts[h] += 1
        return np.sqrt(1.0 / self.counts[h])
