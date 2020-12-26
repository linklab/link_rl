import sys, os
import time
from multiprocessing import Process
import datetime as dt
import torch
print(torch.__version__)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

idx = os.getcwd().index("link_rl")
PROJECT_HOME = os.getcwd()[:idx] + "link_rl"
if PROJECT_HOME not in sys.path:
    sys.path.append(PROJECT_HOME)

from common import common_utils
from config.parameters import PARAMETERS as params
from rl_main import rl_utils
import rl_main.federated_main.utils as utils

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

os.environ["CUDA_VISIBLE_DEVICES"] = params.CUDA_VISIBLE_DEVICES_NUMBER_LIST

if __name__ == "__main__":
    common_utils.print_fast_rl_params(params)

    torch.manual_seed(params.SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(params.SEED)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    utils.check_mqtt_server()
    utils.make_output_folders()
    utils.ask_file_removal(device)

    env = rl_utils.get_environment(params=params)
    rl_model = rl_utils.get_rl_model(env, -1, params=params)

    utils.print_configuration(env, rl_model, params)

    try:
        chief = Process(target=utils.run_chief, args=(params,))
        chief.start()

        time.sleep(1.5)

        workers = []
        for worker_id in range(params.NUM_WORKERS):
            worker = Process(target=utils.run_worker, args=(worker_id, params,))
            workers.append(worker)
            worker.start()

        for worker in workers:
            worker.join()

        chief.join()
    except KeyboardInterrupt as error:
        print("=== {0:>8} is aborted by keyboard interrupt".format('Main'))
