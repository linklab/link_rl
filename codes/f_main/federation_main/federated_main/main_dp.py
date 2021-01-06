import sys, os

import torch

current_path = os.path.dirname(os.path.realpath(__file__))
PROJECT_HOME = os.path.abspath(os.path.join(current_path, os.pardir, os.pardir))

if PROJECT_HOME not in sys.path:
    sys.path.append(PROJECT_HOME)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from codes.a_config.parameters import PARAMETERS as params
from rl_main import rl_utils

env = rl_utils.get_environment(params=params)

if __name__ == "__main__":
    algorithm = rl_utils.get_rl_algorithm(env, params=params)
    state_values, policy, action_table = algorithm.start_iteration()

    print("State Values:\n{0}".format(state_values))
    print()
    print("Policy:\n{0}".format(policy))
    print()
    print("Action Table\n{0}".format(action_table))
    print()
