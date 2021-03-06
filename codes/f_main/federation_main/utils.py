import glob
import math
import os
import numpy as np
import torch
import torch.nn as nn
import sys
import paho.mqtt.client as mqtt

torch.backends.cudnn.benchmark = True

current_path = os.path.dirname(os.path.realpath(__file__))
PROJECT_HOME = os.path.abspath(os.path.join(current_path, os.pardir, os.pardir, os.pardir, os.pardir))
if PROJECT_HOME not in sys.path:
    sys.path.append(PROJECT_HOME)

from codes.a_config.parameters import PARAMETERS as params
from codes.e_utils.names import RLAlgorithmName, DeepLearningModelName


def exp_moving_average(values, window):
    """ Numpy implementation of EMA
    """
    if window >= len(values):
        if len(values) == 0:
            sma = 0.0
        else:
            sma = np.mean(np.asarray(values))
        a = [sma] * len(values)
    else:
        weights = np.exp(np.linspace(-1., 0., window))
        weights /= weights.sum()
        a = np.convolve(values, weights, mode='full')[:len(values)]
        a[:window] = a[window]
    return a


def get_conv2d_size(h, w, kernel_size, padding, stride):
    return math.floor((h - kernel_size + 2 * padding) / stride + 1), math.floor((w - kernel_size + 2 * padding) / stride + 1)


def get_pool2d_size(h, w, kernel_size, stride):
    return math.floor((h - kernel_size) / stride + 1), math.floor((w - kernel_size) / stride + 1)


def print_configuration(env, rl_model, params):
    print("\n*** GENERAL ***")
    print(" NUM WORKERS: {0}".format(params.NUM_WORKERS))
    print(" MODEL SAVE: {0}".format(params.MODEL_SAVE))
    print(" PLATFORM: {0}".format(params.MY_PLATFORM))
    print(" EMA WINDOW: {0}".format(params.EMA_WINDOW))
    print(" SEED: {0}".format(params.SEED))

    print("\n*** MODE ***")
    if params.MODE_SYNCHRONIZATION:
        print(" MODE1: [SYNCHRONOUS_COMMUNICATION] vs. ASYNCHRONOUS_COMMUNICATION")
    else:
        print(" MODE1: SYNCHRONOUS_COMMUNICATION vs. [ASYNCHRONOUS_COMMUNICATION]")

    if params.MODE_GRADIENTS_UPDATE:
        print(" MODE2: [GRADIENTS_UPDATE] vs. NO GRADIENTS_UPDATE")
    else:
        print(" MODE2: GRADIENTS_UPDATE vs. [NO GRADIENTS_UPDATE]")

    if params.MODE_PARAMETERS_TRANSFER:
        print(" MODE3: [PARAMETERS_TRANSFER] vs. NO PARAMETERS_TRANSFER")
    else:
        print(" MODE3: PARAMETERS_TRANSFER vs. [NO PARAMETERS_TRANSFER]")

    print("\n*** MY_PLATFORM & ENVIRONMENT ***")
    print(" Platform: " + params.MY_PLATFORM.value)
    print(" Environment Name: " + params.ENVIRONMENT_ID.value)
    print(" Action Space: {0} - {1}".format(env.get_n_actions(), env.action_meanings))

    print("\n*** RL ALGORITHM ***")
    print(" RL Algorithm: {0}".format(params.RL_ALGORITHM.value))
    if params.RL_ALGORITHM == RLAlgorithmName.CONTINUOUS_PPO_V0:
        print(" PPO_K_EPOCHS: {0}".format(params.PPO_K_EPOCHS))
        print(" PPO_EPSILON_CLIP: {0}".format(params.PPO_EPSILON_CLIP))
        print(" ENTROPY_LOSS_WEIGHT: {0}".format(params.ENTROPY_LOSS_WEIGHT))

    print("\n*** MODEL ***")
    print(" Deep Learning Model: {0}".format(params.DEEP_LEARNING_MODEL.value))
    if params.DEEP_LEARNING_MODEL == DeepLearningModelName.ACTOR_CRITIC_CNN:
        print(" input_width: {0}, input_height: {1}, input_channels: {2}, a_size: {3}, continuous: {4}".format(
            rl_model.input_width,
            rl_model.input_height,
            rl_model.input_channels,
            rl_model.a_size,
            rl_model.continuous
        ))
    elif params.DEEP_LEARNING_MODEL in (
            DeepLearningModelName.ACTOR_CRITIC_MLP,
            DeepLearningModelName.CONTINUOUS_DETERMINISTIC_ACTOR_CRITIC_MLP
    ):
        print(" s_size: {0}, hidden_1: {1}, hidden_2: {2}, hidden_3: {3}, a_size: {4}, continuous: {5}".format(
            rl_model.s_size,
            rl_model.hidden_1_size,
            rl_model.hidden_2_size,
            rl_model.hidden_3_size,
            rl_model.a_size,
            True if params.DEEP_LEARNING_MODEL == DeepLearningModelName.CONTINUOUS_DETERMINISTIC_ACTOR_CRITIC_MLP else False
        ))
    elif params.DEEP_LEARNING_MODEL == DeepLearningModelName.NO_MODEL:
        pass
    else:
        pass

    print("\n*** Optimizer ***")
    print(" Optimizer: {0}".format(params.OPTIMIZER.value))
    print(" Learning Rate: {0}".format(params.LEARNING_RATE))
    print(" Gamma (Discount Factor): {0}".format(params.GAMMA))

    print()
    response = input("Are you OK for All environmental variables? [y/n]: ")
    if not (response == "Y" or response == "y"):
        sys.exit(-1)


def ask_file_removal(device):
    print("CPU/GPU Devices:{0}".format(device))
    response = input("DELETE All Graphs, Logs, and Model Files? [y/n]: ")
    if not (response == "Y" or response == "y"):
        sys.exit(-1)

    files = glob.glob(os.path.join(PROJECT_HOME, "out", "graphs", "*"))
    for f in files:
        os.remove(f)

    files = glob.glob(os.path.join(PROJECT_HOME, "out", "logs", "*"))
    for f in files:
        os.remove(f)

    files = glob.glob(os.path.join(PROJECT_HOME, "out", "out_err", "*"))
    for f in files:
        os.remove(f)

    # files = glob.glob(os.path.join(PROJECT_HOME, "out", "model_save_files", "*"))
    # for f in files:
    #     os.remove(f)

    files = glob.glob(os.path.join(PROJECT_HOME, "out", "save_results", "*"))
    for f in files:
        os.remove(f)


def make_output_folders():
    if not os.path.exists(os.path.join(PROJECT_HOME, "out", "graphs")):
        os.makedirs(os.path.join(PROJECT_HOME, "out", "graphs"))

    if not os.path.exists(os.path.join(PROJECT_HOME, "out", "logs")):
        os.makedirs(os.path.join(PROJECT_HOME, "out", "logs"))

    if not os.path.exists(os.path.join(PROJECT_HOME, "out", "out_err")):
        os.makedirs(os.path.join(PROJECT_HOME, "out", "out_err"))

    if not os.path.exists(os.path.join(PROJECT_HOME, "out", "model_save_files")):
        os.makedirs(os.path.join(PROJECT_HOME, "out", "model_save_files"))

    if not os.path.exists(os.path.join(PROJECT_HOME, "out", "save_results")):
        os.makedirs(os.path.join(PROJECT_HOME, "out", "save_results"))


def check_server_running(host, port):
    try:
        test_mqtt_client = mqtt.Client("test")
        test_mqtt_client.connect(host, port)
        test_mqtt_client.disconnect()
        return True
    except ConnectionRefusedError as e:
        return False


def check_mqtt_server():
    running = check_server_running(params.MQTT_SERVER, params.MQTT_PORT)
    assert running, "MQTT broker is not running at {0}:{1}.\n/usr/local/sbin/mosquitto -c " \
                    "/usr/local/etc/mosquitto/mosquitto.conf".format(params.MQTT_SERVER, params.MQTT_PORT)


def run_chief(params):
    try:
        # with subprocess.Popen([PYTHON_PATH, os.path.join(PROJECT_HOME, "rl_main", "chief_workers", "chief_mqtt_main.py")], shell=False, bufsize=1, stdout=sys.stdout, stderr=sys.stdout) as proc:
        #     output = ""
        #     while True:
        #         # Read line from stdout, break if EOF reached, append line to output
        #         line = proc.stdout.readline()
        #         line = line.decode()
        #         if line == "":
        #             break
        #         output += line
        os.system(params.PYTHON_PATH + " " + os.path.join(PROJECT_HOME, "codes", "f_main", "federation_main", "chief_workers", "chief_mqtt_main.py"))
        sys.stdout = open(os.path.join(PROJECT_HOME, "out", "out_err", "chief_stdout.out"), "wb")
        sys.stderr = open(os.path.join(PROJECT_HOME, "out", "out_err", "chief_stderr.out"), "wb")
    except KeyboardInterrupt:
        sys.stdout.flush()
        sys.stdout.flush()


def run_worker(worker_id, params):
    try:
        os.system(params.PYTHON_PATH + " " + os.path.join(PROJECT_HOME, "codes", "f_main", "federation_main", "chief_workers", "worker_mqtt_main.py") + " {0}".format(worker_id))
        sys.stdout = open(os.path.join(PROJECT_HOME, "out", "out_err", "worker_{0}_stdout.out").format(worker_id), "wb")
        sys.stderr = open(os.path.join(PROJECT_HOME, "out", "out_err", "worker_{0}_stderr.out").format(worker_id), "wb")
    except KeyboardInterrupt:
        sys.stdout.flush()
        sys.stderr.flush()


def util_init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


def print_torch(torch_tensor_name, torch_tensor):
    print("{0}:{1} --> size:{2} --> require_grad:{3}".format(
        torch_tensor_name,
        torch_tensor,
        torch_tensor.size(),
        torch_tensor.requires_grad
    ))


class AddBiases(nn.Module):
    def __init__(self, bias):
        super(AddBiases, self).__init__()
        self._bias = nn.Parameter(bias.unsqueeze(1))

    def forward(self, x):
        if x.dim() == 2:
            bias = self._bias.t().view(1, -1)
        else:
            bias = self._bias.t().view(1, -1, 1, 1)

        return x + bias
