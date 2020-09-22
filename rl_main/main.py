import sys, os
import time
from multiprocessing import Process

import torch

os.environ['KMP_DUPLICATE_LIB_OK']='True'

idx = os.getcwd().index("{0}link_rl".format(os.sep))
PROJECT_HOME = os.getcwd()[:idx+1] + "link_rl{0}".format(os.sep)
sys.path.append(PROJECT_HOME)

from config.parameters import PARAMETERS as params
from rl_main import rl_utils
import rl_main.utils as utils

device = torch.device('cuda' if params.CUDA else 'cpu')

os.environ["CUDA_VISIBLE_DEVICES"] = params.CUDA_VISIBLE_DEVICES_NUMBER_LIST

if __name__ == "__main__":
    torch.manual_seed(params.SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(params.SEED)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    utils.make_output_folders()
    utils.ask_file_removal(device)

    env = rl_utils.get_environment()
    rl_model = rl_utils.get_rl_model(env, -1, params, device)

    utils.print_configuration(env, rl_model, params)

    try:
        chief = Process(target=utils.run_chief, args=(params,))
        chief.start()

        time.sleep(1.5)

        workers = []
        for worker_id in range(params.NUM_WORKERS):
            worker = Process(target=utils.run_worker, args=(worker_id,))
            workers.append(worker)
            worker.start()

        for worker in workers:
            worker.join()

        chief.join()
    except KeyboardInterrupt as error:
        print("=== {0:>8} is aborted by keyboard interrupt".format('Main'))
