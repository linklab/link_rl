from codes.a_config._rl_parameters.on_policy.parameter_ppo import PARAMETERS_PPO
from codes.e_utils.names import *
from codes.a_config.parameters_general import PARAMETERS_GENERAL


class PARAMETERS_FEDERATED_CARTPOLE_PPO_MLP(PARAMETERS_GENERAL, PARAMETERS_PPO):
    # [GENERAL]
    SEED = 1
    MY_PLATFORM = OSName.MAC
    PYTHON_PATH = "~/anaconda/envs/rl/bin/python"

    # [TRANSFER]
    SOFT_TRANSFER = False
    SOFT_TRANSFER_TAU = 0.3

    # [TARGET_UPDATE]
    SOFT_TARGET_UPDATE = False
    SOFT_TARGET_UPDATE_TAU = 0.3

    # [WORKER]
    NUM_WORKERS = 4

    # [OPTIMIZATION]
    MAX_EPISODES = 10000
    GAMMA = 0.98
    #GAMMA = 1.0

    # [MODE]
    MODE_SYNCHRONIZATION = True
    MODE_GRADIENTS_UPDATE = True      # Distributed
    MODE_PARAMETERS_TRANSFER = True    # Transfer

    # [TRAINING]
    EPSILON_INIT = 0.9
    EPSILON_MIN = 0.05
    OPTIMIZER = OptimizerName.ADAM
    PPO_GAE_LAMBDA = 0.99
    LEARNING_RATE = 0.001

    # [TRAJECTORY_SAMPLING]
    TRAJECTORY_SAMPLING = True
    TRAJECTORY_LIMIT_SIZE = 400
    PPO_TRAJECTORY_BATCH_SIZE = 128

    # [PPO]
    PPO_K_EPOCHS = 20
    PPO_EPSILON_CLIP = 0.1

    # [DQN]
    BATCH_SIZE = 128

    # [1. ENVIRONMENTS]
    ENVIRONMENT_ID = EnvironmentName.CARTPOLE_V0

    # [2. DEEP_LEARNING_MODELS]
    DEEP_LEARNING_MODEL = DeepLearningModelName.CONTINUOUS_DETERMINISTIC_ACTOR_CRITIC_MLP

    # [3. ALGORITHMS]
    RL_ALGORITHM = RLAlgorithmName.CONTINUOUS_PPO_V0
