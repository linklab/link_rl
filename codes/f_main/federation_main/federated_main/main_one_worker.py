import sys, os
from multiprocessing import Process
import torch

from codes.f_main.federation_main import utils

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
