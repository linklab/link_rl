import random
from typing import Optional

import gym
import torch

from common.environments.or_gym.envs.classic_or.knapsack import BoundedKnapsackEnv
from common.environments.or_gym.envs.classic_or.tsp import TSPEnv, TSPDistCost
from common.fast_rl.common import wrappers
import numpy as np
import math
from matplotlib import pyplot as plt

from common.fast_rl.common.wrappers import OriginalRewardsWrapper


#https://medium.com/analytics-vidhya/stretched-exponential-decay-function-for-epsilon-greedy-algorithm-98da6224c22f
def stretched_exponential_decay(epsilon_start, epsilon_minimum, epsilon_end_step, current_step):
    end_step = epsilon_end_step
    if current_step < end_step:
        A = 0.5
        B = 0.3
        C = 0.1
        standardized_time = (current_step - A * end_step) / (B * end_step)
        cosh = np.cosh(math.exp(-standardized_time))
        epsilon = epsilon_start - (1 / cosh + (current_step * C / end_step))
        if epsilon >= epsilon_start:
            epsilon = epsilon_start
        elif epsilon <= epsilon_minimum:
            epsilon = epsilon_minimum
    else:
        epsilon = epsilon_minimum
    return epsilon


def print_params(params_class):
    print('\n' + '################ Parameters ################')
    for param in dir(params_class):
        if not param.startswith("__"):
            print("{0}: {1}".format(param, getattr(params_class, param)))
    print('############################################')
    print()


def print_fast_rl_params(params_class):
    print('\n' + '################ Parameters ################')
    for param in dir(params_class):
        if not param.startswith("__") and not param.startswith("MQTT"):
            print("{0}: {1}".format(param, getattr(params_class, param)))
    print('############################################')
    print()


def set_global_seeds(seed):
    myseed = seed + 1000 if seed is not None else None
    try:
        torch.manual_seed(myseed)
        torch.cuda.manual_seed_all(myseed)
    except ImportError:
        pass
    np.random.seed(myseed)
    random.seed(myseed)


def make_atari_env(env_id, rank=0, seed=0):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environment you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    set_global_seeds(seed)

    env = gym.make(env_id)
    env.seed(seed + rank)
    env = wrappers.wrap_dqn(env)
    return env


def make_gym_env(env_id, rank=0, seed=0):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environment you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    set_global_seeds(seed)

    env = gym.make(env_id)
    env = OriginalRewardsWrapper(env)
    env.seed(seed + rank)

    return env


def make_or_gym_env(env_name, env_config, rank=0, seed=0):
    set_global_seeds(seed)
    if env_name == 'Knapsack-v2':
        env = BoundedKnapsackEnv(env_config=env_config)
    elif env_name == "TSP-v0":
        env = TSPEnv(env_config=env_config)
    elif env_name == "TSP-v1":
        env = TSPDistCost(env_config=env_config)
    else:
        raise ValueError(env_name)

    env.set_seed(seed + rank)

    return env


# https://medium.com/analytics-vidhya/stretched-exponential-decay-function-for-epsilon-greedy-algorithm-98da6224c22f
def epsilon_scheduled(current_episode, max_episodes, initial_epsilon, final_epsilon):
    A = 0.05     # spend more time, either on Exploration or on Exploitation
    B = 0.04     # the slope of transition region between Exploration to Exploitation zone
    C = 0.1     # the steepness of left and right tail of the graph
    standardized_time = (current_episode - A * max_episodes)/(B * max_episodes)
    cosh = np.cosh(math.exp(-standardized_time))
    epsilon = max(initial_epsilon - (1 / cosh + (current_episode * C / max_episodes)), final_epsilon)
    return epsilon


def smooth(old: Optional[float], val: float, alpha: float = 0.95) -> float:
    if old is None:
        return val
    return old * alpha + (1.0 - alpha) * val


if __name__=="__main__":
    # max_episode = 20000000
    max_episode = 200
    initial_epsilon = 1.0
    final_epsilon = 0.01
    epsilon_list = []
    plt.plot(
        [x for x in range(max_episode)],
        [epsilon_scheduled(x, max_episode, initial_epsilon, final_epsilon) for x in range(max_episode)]
    )
    plt.show()