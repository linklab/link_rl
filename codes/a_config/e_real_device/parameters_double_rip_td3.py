from codes.a_config._rl_parameters.off_policy.parameter_td3 import PARAMETERS_TD3, TD3ActionType, TD3ActionSelectorType
from codes.a_config.parameters_general import PARAMETERS_GENERAL, RIPEnvRewardType
from codes.e_utils.names import OptimizerName, RLAlgorithmName, EnvironmentName, DeepLearningModelName


class PARAMETERS_DOUBLE_RIP_TD3(PARAMETERS_GENERAL, PARAMETERS_TD3):
    # [GENERAL]
    SEED = 1
    MY_PLATFORM = None
    PYTHON_PATH = None
    EMA_WINDOW = 10
    VERBOSE = True
    MODEL_SAVE = False

    ########################################
    ########################################
    TRAIN_STOP_EPISODE_REWARD = 57000  # MAX: 6.28 * 10000 = 62800 (Old), 4 * 10000 = 40000 (New)
    STOP_PATIENCE_COUNT = 10

    REPLAY_BUFFER_SIZE = 100000
    TARGET_NET_SYNC_STEP_PERIOD = 1000
    MAX_GLOBAL_STEP = 10000000
    TRAIN_STEP_FREQ = 1
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

    # [4. OPTIMIZER]
    OPTIMIZER = OptimizerName.ADAM

    NOISE_ENABLED = True
    OU_SIGMA = 2.5

    COUNT_BASED_EXPLORATION = True
    COUNT_BASED_FILTER = [1, 1, 0, 1, 1, 0, 1, 1, 0]
    COUNT_BASED_REWARD_SCALE = 0.4
    COUNT_BASED_PRECISION = 1

    TRAIN_ONLY_AFTER_EPISODE = False
    NUM_TRAIN_ONLY_AFTER_EPISODE = 100

    TYPE_OF_RIP_REWARD = RIPEnvRewardType.NEW  # "old_version"
    TYPE_OF_TD3_ACTION = TD3ActionType.GAUSSIAN_NOISE_WITH_EPSILON
    TYPE_OF_TD3_ACTION_SELECTOR = TD3ActionSelectorType.BASIC_ACTION_SELECTOR

    DEEP_LEARNING_MODEL = DeepLearningModelName.TD3_MLP
    RL_ALGORITHM = RLAlgorithmName.TD3_V0
