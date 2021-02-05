from codes.b_environments.trade.trade_constant import TimeUnit
from codes.e_utils.names import *
from codes.a_config.parameters_general import PARAMETERS_GENERAL


class PARAMETERS_GENERAL_TRADE_DQN(PARAMETERS_GENERAL):
    ENVIRONMENT_ID = EnvironmentName.TRADE_V0
    DEEP_LEARNING_MODEL = DeepLearningModelName.DUELING_DQN_SMALL_CNN
    RL_ALGORITHM = RLAlgorithmName.DQN_V0
    OPTIMIZER = OptimizerName.ADAM

    STOP_MEAN_EPISODE_REWARD = 250.0
    STOP_PATIENCE_COUNT = 10

    REPLAY_BUFFER_SIZE = 500000
    TARGET_NET_SYNC_STEP_PERIOD = 100
    MAX_GLOBAL_STEP = 10000000
    EPSILON_MIN_STEP = 1500000
    EPSILON_INIT = 1.0
    EPSILON_MIN = 0.01
    LEARNING_RATE = 0.001

    GAMMA = 0.99
    BATCH_SIZE = 32
    TRAIN_STEP_FREQ = 2
    AVG_EPISODE_SIZE_FOR_STAT = 50
    N_STEP = 4

    OMEGA = False
    PER_PROPORTIONAL = False
    PER_RANK_BASED = False
    DOUBLE = True

    COIN_NAME = "BTC"
    TIME_UNIT = TimeUnit.ONE_HOUR