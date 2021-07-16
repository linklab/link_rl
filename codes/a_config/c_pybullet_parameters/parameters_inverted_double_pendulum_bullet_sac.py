from codes.a_config._rl_parameters.off_policy.parameter_sac import PARAMETERS_SAC
from codes.e_utils.names import *
from codes.a_config.parameters_general import PARAMETERS_GENERAL


class PARAMETERS_INVERTED_DOUBLE_PENDULUM_BULLET_SAC(PARAMETERS_GENERAL, PARAMETERS_SAC):
    ENVIRONMENT_ID      = EnvironmentName.PYBULLET_INVERTED_DOUBLE_PENDULUM_V0
    DEEP_LEARNING_MODEL = DeepLearningModelName.CONTINUOUS_SAC_MLP
    RL_ALGORITHM        = RLAlgorithmName.CONTINUOUS_SAC_V0
    OPTIMIZER           = OptimizerName.ADAM

    TRAIN_STOP_EPISODE_REWARD = 9000
    STOP_PATIENCE_COUNT = 10

    TARGET_NET_SYNC_STEP_PERIOD = 10000
    MAX_GLOBAL_STEP = 10000000

    GAMMA = 0.99
    BATCH_SIZE = 128
    AVG_EPISODE_SIZE_FOR_STAT = 50
    N_STEP = 1

