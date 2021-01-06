from codes.e_utils.names import *
from codes.a_config.parameters_general import PARAMETERS_GENERAL


class PARAMETERS_FEDERATED_FROZENLAKE_MONTE_CARLO(PARAMETERS_GENERAL):
    # [GENERAL]
    SEED = 1
    MY_PLATFORM = OSName.MAC
    PYTHON_PATH = "~/anaconda/envs/rl/bin/python"
    ENV_RENDER = False
    MODEL_SAVE = False

    # [WORKER]
    NUM_WORKERS = 1

    # [OPTIMIZATION]
    MAX_EPISODES = 10000
    GAMMA = 0.98

    # [MODE]
    MODE_SYNCHRONIZATION = True
    MODE_GRADIENTS_UPDATE = False      # Distributed
    MODE_PARAMETERS_TRANSFER = False    # Transfer

    # [TRAINING]
    EPSILON_GREEDY_ACT = True
    EPSILON_DECAY = True
    EPSILON_INIT = 0.9
    EPSILON_MIN = 0.05
    EPSILON_DECAY_RATE = 20000 # Large value means low decaying
    LEARNING_RATE = 0.1

    # [1. ENVIRONMENTS]
    ENVIRONMENT_ID = EnvironmentName.FROZENLAKE_V0

    # [2. DEEP_LEARNING_MODELS]
    DEEP_LEARNING_MODEL = DeepLearningModelName.NO_MODEL

    # [3. ALGORITHMS]
    RL_ALGORITHM = RLAlgorithmName.Monte_Carlo_Control_V0