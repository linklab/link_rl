import sys, os
from multiprocessing import Process
import torch

idx = os.getcwd().index("{0}link_rl".format(os.sep))
PROJECT_HOME = os.getcwd()[:idx+1] + "link_rl{0}".format(os.sep)
sys.path.append(PROJECT_HOME)

from config.parameters import PARAMETERS as params

if torch.cuda.is_available():
    device = torch.device("cuda" if params.CUDA else "cpu")
else:
    device = torch.device("cpu")

import rl_main.federated_main.utils as utils

if __name__ == "__main__":
    utils.make_output_folders()
    utils.ask_file_removal(device)

    stderr = sys.stderr
    sys.stderr = sys.stdout

    try:
        # workers = []
        # for worker_id in range(NUM_WORKERS):
        worker = Process(target=utils.run_worker, args=(1,))
            # workers.append(worker)
        worker.start()

        # for worker in workers:
        worker.join()
    except KeyboardInterrupt as error:
        print("=== {0:>8} is aborted by keyboard interrupt".format('Main-Worker'))
    finally:
        sys.stderr = stderr
