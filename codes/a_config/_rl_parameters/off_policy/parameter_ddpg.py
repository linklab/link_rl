import enum

from codes.a_config._rl_parameters.off_policy.parameter_off_policy import PARAMETERS_OFF_POLICY


class DDPGActionSelectorType(enum.Enum):
    BASIC_ACTION_SELECTOR = 0
    SOMETIMES_BLOW_ACTION_SELECTOR = 1
    NOISY_NET_ACTION_SELECTOR = 2


class DDPGActionType(enum.Enum):
    ONLY_OU_NOISE = 0
    OU_NOISE_WITH_EPSILON = 1
    GAUSSIAN_NOISE_WITH_EPSILON = 2
    UNCERTAINTY = 3
    ONLY_GREEDY = 4


class DDPGTrainType(enum.Enum):
    OLD = 0
    NEW = 1


class DDPGTargetUpdateOnlyAfterEpisode(enum.Enum):
    SOFT_UPDATE = 0
    HARD_UPDATE = 1


class PARAMETERS_DDPG(PARAMETERS_OFF_POLICY):
    ENVIRONMENT_ID = None
    PER_PROPORTIONAL = False
    PER_RANK_BASED = False
    DOUBLE = True

    REPLAY_BUFFER_SIZE = None
    TAU = 0.005

    NOISE_ENABLED = True
    OU_SIGMA = 0.2

    EPSILON_INIT = 1.0
    EPSILON_MIN = 0.01
    EPSILON_MIN_STEP = 1000000

    CLIP_GRAD = 3.0

    TRAIN_ONLY_AFTER_EPISODE = False
    NUM_TRAIN_ONLY_AFTER_EPISODE = None

    TYPE_OF_DDPG_ACTION = DDPGActionType.GAUSSIAN_NOISE_WITH_EPSILON  # current
    TYPE_OF_DDPG_TRAIN = DDPGTrainType.NEW  # current
    TYPE_OF_DDPG_ACTION_SELECTOR = DDPGActionSelectorType.BASIC_ACTION_SELECTOR
    TYPE_OF_DDPG_TARGET_UPDATE = DDPGTargetUpdateOnlyAfterEpisode.SOFT_UPDATE

    N_STEP = 2

    NOISY_NET = False

    OMEGA = False