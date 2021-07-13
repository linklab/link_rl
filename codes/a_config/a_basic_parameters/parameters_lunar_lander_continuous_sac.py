from codes.a_config._rl_parameters.off_policy.parameter_sac import PARAMETERS_SAC
from codes.a_config._rl_parameters.on_policy.parameter_ppo import PARAMETERS_PPO
from codes.e_utils.names import *
from codes.a_config.parameters_general import PARAMETERS_GENERAL


class PARAMETERS_LUNAR_LANDER_CONTINUOUS_SAC(PARAMETERS_GENERAL, PARAMETERS_SAC):
    ENVIRONMENT_ID = EnvironmentName.LUNAR_LANDER_CONTINUOUS_V2
    DEEP_LEARNING_MODEL = DeepLearningModelName.CONTINUOUS_SAC_MLP
    RL_ALGORITHM = RLAlgorithmName.CONTINUOUS_SAC_V0
    OPTIMIZER = OptimizerName.ADAM

    TRAIN_STOP_EPISODE_REWARD = 180.0
    STOP_PATIENCE_COUNT = 10

    MAX_GLOBAL_STEP = 10000000
    GAMMA = 0.99
    BATCH_SIZE = 32
    AVG_EPISODE_SIZE_FOR_STAT = 50

    CLIP_GRAD = 3.0

    ACTOR_LEARNING_RATE = 0.0002
    LEARNING_RATE = 0.001
    N_STEP = 2
