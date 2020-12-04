import sys, os
from multiprocessing import Process

import torch

idx = os.getcwd().index("link_rl")
PROJECT_HOME = os.getcwd()[:idx] + "link_rl"
if PROJECT_HOME not in sys.path:
    sys.path.append(PROJECT_HOME)

from config.parameters import PARAMETERS as params
import rl_main.federated_main.utils as utils
from rl_main import rl_utils

if torch.cuda.is_available():
    device = torch.device("cuda" if params.CUDA else "cpu")
else:
    device = torch.device("cpu")


if __name__ == "__main__":
    utils.make_output_folders()
    utils.ask_file_removal(device)

    env = rl_utils.get_environment()
    rl_model = rl_utils.get_rl_model(env)

    utils.print_configuration(env, rl_model, params)

    try:
        chief = Process(target=utils.run_chief, args=(params,))
        chief.start()
        chief.join()
    except KeyboardInterrupt as error:
        print("=== {0:>8} is aborted by keyboard interrupt".format('Main-Chief'))
