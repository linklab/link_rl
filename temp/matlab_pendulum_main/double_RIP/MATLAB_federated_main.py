import sys, os
import time
from multiprocessing import Process
import torch

from codes.e_utils import common_utils, rl_utils
from codes.f_main.federation_main import utils

print(torch.__version__)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

current_path = os.path.dirname(os.path.realpath(__file__))
PROJECT_HOME = os.path.abspath(os.path.join(current_path, os.pardir, os.pardir, os.pardir, os.pardir))
if PROJECT_HOME not in sys.path:
    sys.path.append(PROJECT_HOME)

from codes.a_config.parameters import PARAMETERS as params

os.environ["CUDA_VISIBLE_DEVICES"] = params.CUDA_VISIBLE_DEVICES_NUMBER_LIST

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

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

    input_shape = env.observation_space.shape
    num_outputs = env.action_space.shape[0]
    action_min = env.action_space.low[0]
    action_max = env.action_space.high[0]

    rl_model = rl_utils.get_rl_model(
        worker_id=-1, input_shape=input_shape, num_outputs=num_outputs, params=params, device=device
    )

    #utils.print_configuration(env, rl_model, params)

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
