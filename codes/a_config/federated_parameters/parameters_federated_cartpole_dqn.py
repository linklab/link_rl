from codes.a_config.parameters_general import PARAMETERS_GENERAL
from codes.e_utils.names import EnvironmentName, RLAlgorithmName, DeepLearningModelName, ReplayBufferName, \
    OptimizerName, OSName


class PARAMETERS_FEDERATED_CARTPOLE_DQN(PARAMETERS_GENERAL):
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
    #GAMMA = 1.0

    # [MODE]
    MODE_SYNCHRONIZATION = True
    MODE_GRADIENTS_UPDATE = True      # Distributed
    MODE_PARAMETERS_TRANSFER = True    # Transfer

    # [TRAINING]
    EPSILON_GREEDY_ACT = False
    EPSILON_DECAY = True
    EPSILON_DECAY_RATE = 1000 # Large value means low decaying
    GAE_LAMBDA = 0.99

    # [TRAJECTORY_SAMPLING]
    TRAJECTORY_SAMPLING = True
    TRAJECTORY_LIMIT_SIZE = 400
    TRAJECTORY_BATCH_SIZE = 128

    # [1. ENVIRONMENTS]
    ENVIRONMENT_ID = EnvironmentName.CARTPOLE_V0

    # [2. DEEP_LEARNING_MODELS]
    DEEP_LEARNING_MODEL = DeepLearningModelName.DUELING_DQN_MLP

    # [3. ALGORITHMS]
    RL_ALGORITHM = RLAlgorithmName.DQN_FAST_V0

    # [4. OPTIMIZER]
    OPTIMIZER = OptimizerName.ADAM

    NEXT_STATE_IN_TRAJECTORY = False

    STOP_MEAN_EPISODE_REWARD = 20
    AVG_EPISODE_SIZE_FOR_STAT = 10
    REPLAY_BUFFER_SIZE = 500000
    MAX_GLOBAL_STEP = 100000
    LEARNING_RATE = 0.001
    ACTOR_LEARNING_RATE = 0.0001
    TRAIN_STEP_FREQ = 1
    GAMMA = 0.99
    BATCH_SIZE = 64
    MODEL_SAVE_STEP_PERIOD = 100000
    MIN_REPLAY_SIZE_FOR_TRAIN = 500
    DRAW_VIZ = False
    N_STEP = 4
    OMEGA = False

    EPSILON_INIT = 0.9
    EPSILON_MIN = 0.001
    EPSILON_MIN_STEP = 10000

    CUDA = False

    CLIP_GRAD = 0.1

    PER = False

    ENV_RENDER = False

