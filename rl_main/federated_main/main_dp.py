import sys, os

import torch

idx = os.getcwd().index("link_rl")
PROJECT_HOME = os.getcwd()[:idx] + "link_rl"
if PROJECT_HOME not in sys.path:
    sys.path.append(PROJECT_HOME)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from rl_main import rl_utils

env = rl_utils.get_environment()

if __name__ == "__main__":
    algorithm = rl_utils.get_rl_algorithm(env)
    state_values, policy, action_table = algorithm.start_iteration()

    print("State Values:\n{0}".format(state_values))
    print()
    print("Policy:\n{0}".format(policy))
    print()
    print("Action Table\n{0}".format(action_table))
    print()
