from codes.a_config._rl_parameters.off_policy.parameter_ddpg import PARAMETERS_DDPG
from codes.a_config.parameters_general import PARAMETERS_GENERAL
from codes.e_utils.names import OptimizerName, RLAlgorithmName, EnvironmentName, DeepLearningModelName, ModelSaveMode
from codes.a_config.parameters_general import PARAMETERS_GENERAL

class PARAMETERS_DOUBLE_RIP(PARAMETERS_GENERAL, PARAMETERS_DDPG):
    # [GENERAL]
    SEED = 1
    MY_PLATFORM = None
    PYTHON_PATH = None
    EMA_WINDOW = 10
    VERBOSE = True
    MODEL_SAVE = False

    ########################################
    ########################################
    TRAIN_STOP_EPISODE_REWARD = 100000
    STOP_PATIENCE_COUNT = 10

    REPLAY_BUFFER_SIZE = 100000
    TARGET_NET_SYNC_STEP_PERIOD = 1000
    MAX_GLOBAL_STEP = 10000000
    TRAIN_STEP_FREQ = 4
    AVG_EPISODE_SIZE_FOR_STAT = 10
    N_STEP = 4
    OMEGA = False
    OMEGA_WINDOW_SIZE = 6
    NEXT_STATE_IN_TRAJECTORY = True
    DATA_SAVE_STEP_PERIOD = 1000

    #########################################
    #########################################

    # [MLP_DEEP_LEARNING_MODEL]
    HIDDEN_1_SIZE = 128
    HIDDEN_2_SIZE = 128
    HIDDEN_3_SIZE = 128
    # HIDDEN_SIZE_LIST = [128, 128, 128, 256]

    # [OPTIMIZATION]
    GAMMA = 0.98 # discount factor
    TAU = 0.0001

    # [Policy Gradient]
    ENTROPY_LOSS_WEIGHT = 0.0001
    CLIP_GRAD = 0.1
    ACTOR_LEARNING_RATE = 0.0002

    # [TRAINING]
    EPSILON_INIT = 1.0  # 0.9
    EPSILON_MIN = 1.0   # 0.001
    EPSILON_MIN_STEP = 1000000

    PER_PROPORTIONAL = False
    PER_RANK_BASED = False
    PPO_GAE_LAMBDA = 0.95
    LEARNING_RATE = 0.001
    ACTION_SCALE = 400
    BALANCING_SCALE_FACTOR = 0.01
    ENV_RESET = False

    # [DQN]
    BATCH_SIZE = 128    # 32

    # [CUDA]
    CUDA_VISIBLE_DEVICES_NUMBER_LIST = '1, 2'

    # [1. ENVIRONMENTS]
    ENVIRONMENT_ID = EnvironmentName.REAL_DEVICE_DOUBLE_RIP

    # [2. DEEP_LEARNING_MODELS]
    DEEP_LEARNING_MODEL = DeepLearningModelName.DETERMINISTIC_CONTINUOUS_ACTOR_CRITIC_MLP

    # [3. ALGORITHMS]
    RL_ALGORITHM = RLAlgorithmName.DDPG_V0

    # [4. OPTIMIZER]
    OPTIMIZER = OptimizerName.ADAM

    MODEL_SAVE_MODE = ModelSaveMode.TRAIN

    OU_NOISE_ENABLED = True
    OU_SIGMA = 3.0
    COUNT_BASED_EXPLORATION = True
    COUNT_BASED_REWARD_SCALE = 0.4
    COUNT_BASED_PRECISION = 1
