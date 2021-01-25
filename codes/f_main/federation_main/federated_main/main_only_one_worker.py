import sys, os
from multiprocessing import Process
import torch

from codes.f_main.federation_main.federated_main import utils

current_path = os.path.dirname(os.path.realpath(__file__))
PROJECT_HOME = os.path.abspath(os.path.join(current_path, os.pardir, os.pardir))

if PROJECT_HOME not in sys.path:
    sys.path.append(PROJECT_HOME)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
