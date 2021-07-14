from codes.a_config._rl_parameters.on_policy.parameter_ppo import PARAMETERS_PPO
from codes.e_utils.names import *
from codes.a_config.parameters_general import PARAMETERS_GENERAL


class PARAMETERS_FEDERATED_PENDULUM_PPO(PARAMETERS_GENERAL, PARAMETERS_PPO):
    # [GENERAL]
    SEED = 1
    MY_PLATFORM = OSName.MAC
    PYTHON_PATH = "~/anaconda/envs/rl/bin/python"

    # [TRANSFER]
    SOFT_TRANSFER = True
    SOFT_TRANSFER_TAU = 0.3

    # [TARGET_UPDATE]
    SOFT_TARGET_UPDATE = False
    SOFT_TARGET_UPDATE_TAU = 0.3

    # [WORKER]
    NUM_WORKERS = 4

    # [MODE]
    MODE_SYNCHRONIZATION = True
    MODE_GRADIENTS_UPDATE = True      # Distributed
    MODE_PARAMETERS_TRANSFER = False    # Transfer

    # [1. ENVIRONMENTS]
    ENVIRONMENT_ID = EnvironmentName.PENDULUM_V0

    # [2. DEEP_LEARNING_MODELS]
    DEEP_LEARNING_MODEL = DeepLearningModelName.ACTOR_CRITIC_MLP

    # [3. ALGORITHMS]
    RL_ALGORITHM = RLAlgorithmName.CONTINUOUS_PPO_V0

    # [4. OPTIMIZER]
    OPTIMIZER = OptimizerName.ADAM

    TRAIN_STOP_EPISODE_REWARD = -110
    STOP_PATIENCE_COUNT = 10
    AVG_EPISODE_SIZE_FOR_STAT = 50

    MAX_GLOBAL_STEP = 500000
    LEARNING_RATE = 0.001
    ACTOR_LEARNING_RATE = 0.00001
    GAMMA = 0.99
    BATCH_SIZE = 64

    N_STEP = 4
    OMEGA = False

    EPSILON_INIT = 0.9
    EPSILON_MIN = 0.001
    EPSILON_MIN_STEP = 100000

    CLIP_GRAD = 3.0

    ACTION_SCALE = 2.0
