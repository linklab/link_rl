import enum

from codes.a_config._rl_parameters.off_policy.parameter_off_policy import PARAMETERS_OFF_POLICY


class DQNActionSelectorType(enum.Enum):
    BASIC_ACTION_SELECTOR = 0
    SOMETIMES_BLOW_ACTION_SELECTOR = 1
    NOISY_NET_ACTION_SELECTOR = 2


class DQNActionType(enum.Enum):
    EPSILON_GREEDY = 0
    ONLY_GREEDY = 1


class PARAMETERS_DQN(PARAMETERS_OFF_POLICY):
    PER_PROPORTIONAL = False
    PER_RANK_BASED = False
    DOUBLE = True

    EPSILON_INIT = None
    EPSILON_MIN = None
    EPSILON_MIN_STEP = None

    REPLAY_BUFFER_SIZE = None

    OMEGA = False
    OMEGA_WINDOW_SIZE = None

    NOISY_NET = False
    TRAIN_ONLY_AFTER_EPISODE = False

    DISTRIBUTIONAL = False
    NUM_SUPPORTS = 51
    VALUE_MIN = -10
    VALUE_MAX = 10

    TARGET_NET_SYNC_STEP_PERIOD = 200

    TYPE_OF_DQN_ACTION = DQNActionType.EPSILON_GREEDY
    TYPE_OF_DQN_ACTION_SELECTOR = DQNActionSelectorType.BASIC_ACTION_SELECTOR
