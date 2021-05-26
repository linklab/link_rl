from codes.e_utils.names import *
from codes.a_config.parameters_general import PARAMETERS_GENERAL


class PARAMETERS_FEDERATED_DP(PARAMETERS_GENERAL):
    # [GENERAL]
    SEED = 1
    MY_PLATFORM = OSName.MAC
    PYTHON_PATH = "~/anaconda/envs/rl/bin/python"
    MODEL_SAVE = False

    # [OPTIMIZATION]
    MAX_EPISODES = 5000
    GAMMA = 0.98

    # [1. ENVIRONMENTS]
    ENVIRONMENT_ID = EnvironmentName.FROZENLAKE_V0

    # [2. DEEP_LEARNING_MODELS]
    DEEP_LEARNING_MODEL = DeepLearningModelName.NO_MODEL

    # [3. ALGORITHMS]
    RL_ALGORITHM = RLAlgorithmName.Policy_Iteration