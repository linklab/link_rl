import sys, os
from multiprocessing import Process

import torch

idx = os.getcwd().index("{0}link_rl".format(os.sep))
PROJECT_HOME = os.getcwd()[:idx+1] + "link_rl{0}".format(os.sep)
sys.path.append(PROJECT_HOME)

from config.parameters import PARAMETERS as params
import rl_main.utils as utils
from rl_main import rl_utils

device = torch.device('cuda' if params.CUDA else 'cpu')


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
