from codes.a_config._rl_parameters.off_policy.parameter_dqn import PARAMETERS_DQN
from codes.e_utils.names import *
from codes.a_config.parameters_general import PARAMETERS_GENERAL


class PARAMETERS_CARTPOLE_RAINBOW(PARAMETERS_GENERAL, PARAMETERS_DQN):
    ENVIRONMENT_ID = EnvironmentName.CARTPOLE_V1
    RL_ALGORITHM = RLAlgorithmName.RAINBOW_V0
    DEEP_LEARNING_MODEL = DeepLearningModelName.RAINBOW_DQN_MLP

    TRAIN_STOP_EPISODE_REWARD = 395.0
    STOP_PATIENCE_COUNT = 10
    REPLAY_BUFFER_SIZE = 50000
    TARGET_NET_SYNC_STEP_PERIOD = 100
    MAX_GLOBAL_STEP = 30000
    EPSILON_MIN_STEP = 5000
    EPSILON_INIT = 1.0
    EPSILON_MIN = 0.01
    LEARNING_RATE = 0.001
    GAMMA = 0.99
    BATCH_SIZE = 32
    TRAIN_STEP_FREQ = 2
    AVG_EPISODE_SIZE_FOR_STAT = 50
    N_STEP = 4
    OMEGA = False
