import sys, os
from multiprocessing import Process

import torch

from codes.a_config.parameters import PARAMETERS as params
import codes.f_main.federation_main.utils as utils
from rl_main import rl_utils

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


if __name__ == "__main__":
    utils.make_output_folders()
    utils.ask_file_removal(device)

    env = rl_utils.get_environment(params=params)
    rl_model = rl_utils.get_rl_model(env, params=params)

    utils.print_configuration(env, rl_model, params)

    try:
        chief = Process(target=utils.run_chief, args=(params,))
        chief.start()
        chief.join()
    except KeyboardInterrupt as error:
        print("=== {0:>8} is aborted by keyboard interrupt".format('Main-Chief'))
