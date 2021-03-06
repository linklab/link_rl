from codes.a_config._rl_parameters.off_policy.parameter_dqn import PARAMETERS_DQN, DQNActionType, DQNActionSelectorType
from codes.a_config._rl_parameters.off_policy.parameter_td3 import PARAMETERS_TD3, TD3ActionType, TD3ActionSelectorType
from codes.a_config.e_real_device.parameters_quanser_rip_td3 import PARAMETERS_QUANSER_RIP_TD3
from codes.e_utils.names import *
from codes.a_config.parameters_general import PARAMETERS_GENERAL
import math

class PARAMETERS_ADJUST_ANGLE_DQN(PARAMETERS_GENERAL, PARAMETERS_DQN):
    ENVIRONMENT_ID = EnvironmentName.ADJUST_ANGLE_V0
    RL_ALGORITHM = RLAlgorithmName.DQN_V0
    DEEP_LEARNING_MODEL = DeepLearningModelName.DUELING_DQN_MLP

    VELOCITY_STATE_DENOMINATOR = 100
    REWARD_DENOMINATOR = math.pi

    TRAIN_STOP_EPISODE_REWARD = 900  # MAX: 3.14 * 10000 = 31400
    TRAIN_STOP_EPISODE_REWARD_STD = 50.0
    STOP_PATIENCE_COUNT = 50
    REPLAY_BUFFER_SIZE = 1000000

    MAX_EPISODE_STEP_AT_PLAY = 1000
    MAX_EPISODE_STEP = 1000
    MAX_GLOBAL_STEP = 10000000
    LEARNING_RATE = 0.001
    ACTOR_LEARNING_RATE = 0.0002
    GAMMA = 0.99

    BATCH_SIZE = 128
    AVG_EPISODE_SIZE_FOR_STAT = 50

    N_STEP = 4
    OMEGA = False
    PER_PROPORTIONAL = False
    PER_RANK_BASED = False

    TEST_NUM_EPISODES = 3

    DISTRIBUTIONAL = False
    NOISY_NET = False

    EPSILON_INIT = 1.0
    EPSILON_MIN = 0.01
    EPSILON_MIN_STEP = 3000000

    TYPE_OF_DQN_ACTION = DQNActionType.EPSILON_GREEDY
    TYPE_OF_DQN_ACTION_SELECTOR = DQNActionSelectorType.SOMETIMES_BLOW_ACTION_SELECTOR

    TRAIN_STEP_FREQ = 2

    GOAL_ANGLE = 0.0
    UNIT_TIME = 0.006

    VERBOSE_TO_LOG = False