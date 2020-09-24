import gym

from common.fast_rl.common import wrappers


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
