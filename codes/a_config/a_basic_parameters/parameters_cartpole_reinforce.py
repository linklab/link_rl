from codes.a_config._rl_parameters.off_policy.parameter_dqn import PARAMETERS_DQN
from codes.e_utils.names import *
from codes.a_config.parameters_general import PARAMETERS_GENERAL


class PARAMETERS_CARTPOLE_REINFORCE(PARAMETERS_GENERAL, PARAMETERS_DQN):
    ENVIRONMENT_ID = EnvironmentName.CARTPOLE_V1
    RL_ALGORITHM = RLAlgorithmName.REINFORCE_V0
    DEEP_LEARNING_MODEL = DeepLearningModelName.DUELING_DQN_MLP

    TRAIN_STOP_EPISODE_REWARD = 195.0
    STOP_PATIENCE_COUNT = 10

    MAX_GLOBAL_STEP = 100000
    LEARNING_RATE = 0.001
    GAMMA = 0.99
    BATCH_SIZE = 32
    AVG_EPISODE_SIZE_FOR_STAT = 50
    N_STEP = 10
    OMEGA = False

    EPISODES_TO_TRAIN = 4

