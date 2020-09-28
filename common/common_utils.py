import gym

from common.fast_rl.common import wrappers
import numpy as np
import math
from matplotlib import pyplot as plt

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


def make_atari_env(params):
    env = gym.make(params.ENVIRONMENT_ID.value)
    env = wrappers.wrap_dqn(env)
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
