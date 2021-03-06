import sys, os
import time
from multiprocessing import Process
import torch

from codes.e_utils import common_utils
from codes.f_main.federation_main import utils

print(torch.__version__)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

current_path = os.path.dirname(os.path.realpath(__file__))
PROJECT_HOME = os.path.abspath(os.path.join(current_path, os.pardir, os.pardir, os.pardir, os.pardir))
if PROJECT_HOME not in sys.path:
    sys.path.append(PROJECT_HOME)

from codes.a_config.parameters import PARAMETERS as params

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

    # env = rl_utils.get_environment(params=params)
    # rl_model = rl_utils.get_rl_model(env, -1, params=params)
    #
    # utils.print_configuration(env, rl_model, params)

    try:
        processes = []

        chief = Process(target=utils.run_chief, args=(params,))
        chief.start()
        processes.append(chief)
        time.sleep(1.5)

        for worker_id in range(params.NUM_WORKERS):
            worker = Process(target=utils.run_worker, args=(worker_id, params,))
            processes.append(worker)
            worker.start()

        for process in processes:
            process.join()

    except KeyboardInterrupt as error:
        print("=== {0:>8} is aborted by keyboard interrupt".format('Main'))
        # env.stop()
