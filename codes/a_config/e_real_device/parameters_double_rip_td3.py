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
    NUM_ENVIRONMENTS = 1

    ########################################
    ########################################
    VELOCITY_STATE_DENOMINATOR = 500.0
    REWARD_DENOMINATOR = 9.0

    TRAIN_STOP_EPISODE_REWARD = 7000  # MAX: 6.28 * 5000 = 62800 (Old), 90000 (New)
    TRAIN_STOP_EPISODE_REWARD_STD = 100
    STOP_PATIENCE_COUNT = 10

    REPLAY_BUFFER_SIZE = 5000000
    AVG_EPISODE_SIZE_FOR_STAT = 10
    N_STEP = 2

    #########################################
    #########################################

    # [MLP_DEEP_LEARNING_MODEL]
    HIDDEN_1_SIZE = 128
    HIDDEN_2_SIZE = 128
    HIDDEN_3_SIZE = 128
    # HIDDEN_SIZE_LIST = [128, 128, 128, 256]

    # [OPTIMIZATION]
    GAMMA = 0.99 # discount factor
    TAU = 0.0001

    # [Policy Gradient]
    CLIP_GRAD = 0.1
    ACTOR_LEARNING_RATE = 0.0002

    # [TRAINING]
    EPSILON_INIT = 1.0  # 0.9
    EPSILON_MIN = 0.01   # 0.001

    LEARNING_RATE = 0.001
    ACTION_SCALE = 350
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

    TRAIN_ONLY_AFTER_EPISODE = False
    NUM_TRAIN_ONLY_AFTER_EPISODE = 100

    DISTRIBUTIONAL = False

    TYPE_OF_RIP_REWARD = RIPEnvRewardType.NEW  # "old_version"
    TYPE_OF_TD3_ACTION = TD3ActionType.GAUSSIAN_NOISE
    TYPE_OF_TD3_ACTION_SELECTOR = TD3ActionSelectorType.SOMETIMES_BLOW_ACTION_SELECTOR

    DEEP_LEARNING_MODEL = DeepLearningModelName.TD3_MLP
    RL_ALGORITHM = RLAlgorithmName.TD3_V0

    TRAIN_STEP_FREQ = 4
    POLICY_UPDATE_FREQUENCY = 2 * TRAIN_STEP_FREQ

    MAX_EPISODE_STEP_AT_PLAY = 10000000000
    MAX_EPISODE_STEP = 10000
    MAX_GLOBAL_STEP = 40000000
    EPSILON_MIN_STEP = 10000000

    DOUBLE_PENDULUM_STATE_INFO = 2  # 1: ALL ARM INFO IGNORED, 2: ARM ANGLE INFO IGNORED (VELOCITY IS INCLUDED)

    UNIT_TIME = 0.006

    PERIODIC_MODEL_SAVE = True

