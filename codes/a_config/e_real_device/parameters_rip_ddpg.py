from codes.a_config._rl_parameters.off_policy.parameter_ddpg import PARAMETERS_DDPG, DDPGActionType, \
    DDPGActionSelectorType
from codes.a_config.parameters_general import RIPEnvRewardType
from codes.e_utils.names import OptimizerName, RLAlgorithmName, EnvironmentName, DeepLearningModelName
from codes.a_config.parameters_general import PARAMETERS_GENERAL

class PARAMETERS_RIP_DDPG(PARAMETERS_GENERAL, PARAMETERS_DDPG):
    # [GENERAL]
    SEED = 1
    MY_PLATFORM = None
    PYTHON_PATH = None
    EMA_WINDOW = 10
    VERBOSE = True
    MODEL_SAVE = False

    ########################################
    ########################################
    TRAIN_STOP_EPISODE_REWARD = 500000  # MAX: 6.28 * 5000 = 62800 (Old), 90000 (New)
    TRAIN_STOP_EPISODE_REWARD_STD = 2000
    STOP_PATIENCE_COUNT = 10

    REPLAY_BUFFER_SIZE = 100000
    TARGET_NET_SYNC_STEP_PERIOD = 1000

    TRAIN_STEP_FREQ = 2
    AVG_EPISODE_SIZE_FOR_STAT = 10
    N_STEP = 4
    OMEGA = False
    OMEGA_WINDOW_SIZE = 6

    #########################################
    #########################################

    # [MLP_DEEP_LEARNING_MODEL]
    HIDDEN_1_SIZE = 128
    HIDDEN_2_SIZE = 128
    HIDDEN_3_SIZE = 128
    # HIDDEN_SIZE_LIST = [128, 128, 128, 256]

    # [OPTIMIZATION]
    GAMMA = 0.999 # discount factor
    TAU = 0.0001

    # [Policy Gradient]
    ENTROPY_LOSS_WEIGHT = 0.0001
    CLIP_GRAD = 0.1
    ACTOR_LEARNING_RATE = 0.0002

    # [TRAINING]
    EPSILON_INIT = 1.0  # 0.9
    EPSILON_MIN = 0.01   # 0.001

    PER_PROPORTIONAL = False
    PER_RANK_BASED = False
    PPO_GAE_LAMBDA = 0.95
    LEARNING_RATE = 0.001
    ACTION_SCALE = 500
    BALANCING_SCALE_FACTOR = 0.01
    ENV_RESET = False

    # [DQN]
    BATCH_SIZE = 128    # 32

    # [CUDA]
    CUDA_VISIBLE_DEVICES_NUMBER_LIST = '1, 2'

    # [1. ENVIRONMENTS]
    ENVIRONMENT_ID = EnvironmentName.REAL_DEVICE_RIP

    # [2. DEEP_LEARNING_MODELS]
    DEEP_LEARNING_MODEL = DeepLearningModelName.DETERMINISTIC_CONTINUOUS_ACTOR_CRITIC_MLP

    # [3. ALGORITHMS]
    RL_ALGORITHM = RLAlgorithmName.DDPG_V0

    # [4. OPTIMIZER]
    OPTIMIZER = OptimizerName.ADAM

    NOISE_ENABLED = True
    OU_SIGMA = 2.5

    TRAIN_ONLY_AFTER_EPISODE = False
    NUM_TRAIN_ONLY_AFTER_EPISODE = 100

    TYPE_OF_DDPG_ACTION = DDPGActionType.GAUSSIAN_NOISE_WITH_EPSILON
    TYPE_OF_DDPG_ACTION_SELECTOR = DDPGActionSelectorType.SOMETIMES_BLOW_ACTION_SELECTOR

    MAX_EPISODE_STEP = 5000
    MAX_GLOBAL_STEP = 10000000
    EPSILON_MIN_STEP = 2000000
    MIN_REPLAY_SIZE_FOR_TRAIN = 500

    IGNORE_ARM_INFO = False

    UNIT_TIME = 0.006

