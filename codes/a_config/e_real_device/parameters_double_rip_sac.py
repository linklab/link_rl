from codes.a_config._rl_parameters.off_policy.parameter_sac import PARAMETERS_SAC
from codes.a_config._rl_parameters.off_policy.parameter_td3 import PARAMETERS_TD3, TD3ActionType, TD3ActionSelectorType
from codes.a_config.parameters_general import PARAMETERS_GENERAL, RIPEnvRewardType, StochasticActionType, \
    StochasticActionSelectorType
from codes.e_utils.names import OptimizerName, RLAlgorithmName, EnvironmentName, DeepLearningModelName


class PARAMETERS_DOUBLE_RIP_SAC(PARAMETERS_GENERAL, PARAMETERS_SAC):
    ENVIRONMENT_ID = EnvironmentName.REAL_DEVICE_DOUBLE_RIP
    RL_ALGORITHM = RLAlgorithmName.CONTINUOUS_SAC_V0
    DEEP_LEARNING_MODEL = DeepLearningModelName.CONTINUOUS_SAC_MLP
    OPTIMIZER = OptimizerName.ADAM

    ########################################
    ########################################
    VELOCITY_STATE_DENOMINATOR = 500.0
    REWARD_DENOMINATOR = 9.0

    TRAIN_STOP_EPISODE_REWARD = 700  # MAX: 6.28 * 5000 = 62800 (Old), 90000 (New)
    TRAIN_STOP_EPISODE_REWARD_STD = 100
    STOP_PATIENCE_COUNT = 10

    REPLAY_BUFFER_SIZE = 5000000
    AVG_EPISODE_SIZE_FOR_STAT = 50
    N_STEP = 2

    #########################################
    #########################################

    # [OPTIMIZATION]
    GAMMA = 0.99 # discount factor
    TAU = 0.001

    # [Policy Gradient]
    CLIP_GRAD = 3.0
    ACTOR_LEARNING_RATE = 0.0001

    # [TRAINING]
    LEARNING_RATE = 0.001
    ACTION_SCALE = 500
    BALANCING_SCALE_FACTOR = 0.01

    ENV_RESET = False

    # [DQN]
    BATCH_SIZE = 128    # 32

    # [CUDA]
    CUDA_VISIBLE_DEVICES_NUMBER_LIST = '1, 2'

    TRAIN_ONLY_AFTER_EPISODE = False
    NUM_TRAIN_ONLY_AFTER_EPISODE = 100

    DISTRIBUTIONAL = False

    TYPE_OF_RIP_REWARD = RIPEnvRewardType.NEW  # "old_version"

    TYPE_OF_STOCHASTIC_ACTION = StochasticActionType.SAMPLE
    TYPE_OF_STOCHASTIC_ACTION_SELECTOR = StochasticActionSelectorType.SOMETIMES_BLOW_ACTION_SELECTOR

    TRAIN_STEP_FREQ = 4
    POLICY_UPDATE_FREQUENCY = 2 * TRAIN_STEP_FREQ

    MAX_EPISODE_STEP_AT_PLAY = 10000000000
    MAX_EPISODE_STEP = 1000
    MAX_GLOBAL_STEP = 40000000
    EPSILON_MIN_STEP = 10000000

    DOUBLE_PENDULUM_STATE_INFO = 2  # 1: ALL ARM INFO IGNORED, 2: ARM ANGLE INFO IGNORED (VELOCITY IS INCLUDED)

    UNIT_TIME = 0.007

    PERIODIC_MODEL_SAVE = True

    VERBOSE_TO_LOG = True

