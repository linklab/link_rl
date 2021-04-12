from codes.a_config._rl_parameters.off_policy.parameter_ddpg import PARAMETERS_DDPG
from codes.a_config._rl_parameters.off_policy.parameter_td3 import PARAMETERS_TD3
from codes.e_utils.names import *
from codes.a_config.parameters_general import PARAMETERS_GENERAL


class PARAMETERS_PENDULUM_MATLAB_TD3(PARAMETERS_GENERAL, PARAMETERS_TD3):
    ENV_RESET = True

    ENVIRONMENT_ID = EnvironmentName.PENDULUM_MATLAB_V0
    RL_ALGORITHM = RLAlgorithmName.TD3_V0
    DEEP_LEARNING_MODEL = DeepLearningModelName.TD3_MLP

    TRAIN_STOP_EPISODE_REWARD = 75000  # MAX: 6.28 * 10000 = 62800 (Old), 4 * 10000 = 40000 (New)
    TRAIN_STOP_EPISODE_REWARD_STD = 1000

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
    ACTION_SCALE = 1.8

    MODEL_SAVE_MODE = ModelSaveMode.TEST

    OU_NOISE_ENABLED = True
    OU_SIGMA = 2.0

    COUNT_BASED_EXPLORATION = True
    COUNT_BASED_FILTER = [1, 1, 1, 1, 1, 0, 1, 1, 0]
    COUNT_BASED_REWARD_SCALE = 0.4
    COUNT_BASED_PRECISION = 0

    TRAIN_ONLY_AFTER_EPISODE = True
    NUM_TRAIN_ONLY_AFTER_EPISODE = 100

    TYPE_OF_ACTION = "old"
    TYPE_OF_REWARD = "current_version"  # "old_version"
    TYPE_OF_TD3_ACTION_SELECTOR = "SomeTimesBlowTD3ActionSelector"

    EPSILON_INIT = 1.0
    EPSILON_MIN = 0.01
    EPSILON_MIN_STEP = 1000000


