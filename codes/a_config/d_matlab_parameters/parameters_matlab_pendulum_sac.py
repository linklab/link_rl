from codes.a_config._rl_parameters.off_policy.parameter_sac import PARAMETERS_SAC, SACActionType, SACActionSelectorType
from codes.a_config._rl_parameters.off_policy.parameter_td3 import PARAMETERS_TD3, TD3ActionType, TD3ActionSelectorType
from codes.e_utils.names import *
from codes.a_config.parameters_general import PARAMETERS_GENERAL, RIPEnvRewardType


class PARAMETERS_PENDULUM_MATLAB_SAC(PARAMETERS_GENERAL, PARAMETERS_SAC):
    ENV_RESET = True

    ENVIRONMENT_ID = EnvironmentName.PENDULUM_MATLAB_V0
    RL_ALGORITHM = RLAlgorithmName.SAC_V0
    DEEP_LEARNING_MODEL = DeepLearningModelName.SOFT_ACTOR_CRITIC_MLP

    TRAIN_STOP_EPISODE_REWARD = 40000  # MAX: 6.28 * 10000 = 62800 (Old), 90000 (New)
    TRAIN_STOP_EPISODE_REWARD_STD = 2000

    STOP_PATIENCE_COUNT = 10
    REPLAY_BUFFER_SIZE = 500000
    MAX_GLOBAL_STEP = 10000000

    LEARNING_RATE = 0.002
    ACTOR_LEARNING_RATE = 0.0002

    TRAIN_STEP_FREQ = 4
    GAMMA = 0.999
    BATCH_SIZE = 128
    AVG_EPISODE_SIZE_FOR_STAT = 50

    N_STEP = 2

    CLIP_GRAD = 0.1
    ACTION_SCALE = 2.5

    TRAIN_ONLY_AFTER_EPISODE = False
    NUM_TRAIN_ONLY_AFTER_EPISODE = 100

    TYPE_OF_RIP_REWARD = RIPEnvRewardType.NEW  # "old_version"

    TYPE_OF_SAC_ACTION = SACActionType.GAUSSIAN_NOISE
    TYPE_OF_SAC_ACTION_SELECTOR = SACActionSelectorType.SOMETIMES_BLOW_ACTION_SELECTOR

