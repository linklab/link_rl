from codes.a_config._rl_parameters.off_policy.parameter_dqn import PARAMETERS_DQN
from codes.e_utils.names import *
from codes.a_config.parameters_general import PARAMETERS_GENERAL


class PARAMETERS_CARTPOLE_DQN(PARAMETERS_GENERAL, PARAMETERS_DQN):
    ENVIRONMENT_ID = EnvironmentName.CARTPOLE_V1
    DEEP_LEARNING_MODEL = DeepLearningModelName.DUELING_DQN_MLP
    RL_ALGORITHM = RLAlgorithmName.DQN_V0
    OPTIMIZER = OptimizerName.ADAM

    TRAIN_STOP_EPISODE_REWARD = 450.0
    STOP_PATIENCE_COUNT = 10
    REPLAY_BUFFER_SIZE = 50000
    TARGET_NET_SYNC_STEP_PERIOD = 100
    MAX_GLOBAL_STEP = 200000

    EPSILON_INIT = 1.0
    EPSILON_MIN = 0.01
    EPSILON_MIN_STEP = 30000

    LEARNING_RATE = 0.001

    GAMMA = 0.99
    BATCH_SIZE = 32
    TRAIN_STEP_FREQ = 2
    AVG_EPISODE_SIZE_FOR_STAT = 50
    N_STEP = 1

    OMEGA = False
    PER_PROPORTIONAL = False
    PER_RANK_BASED = False
    DOUBLE = True
