# https://towardsdatascience.com/learning-to-play-cartpole-and-lunarlander-with-proximal-policy-optimization-dacbd6045417
from codes.e_utils.names import *
from codes.a_config.parameters_general import PARAMETERS_GENERAL


class PARAMETERS_LUNAR_LANDER_A2C(PARAMETERS_GENERAL):
    ENVIRONMENT_ID = EnvironmentName.LUNAR_LANDER_V2
    DEEP_LEARNING_MODEL = DeepLearningModelName.STOCHASTIC_DISCRETE_ACTOR_CRITIC_MLP
    RL_ALGORITHM = RLAlgorithmName.DISCRETE_A2C_V0
    OPTIMIZER = OptimizerName.ADAM

    STOP_MEAN_EPISODE_REWARD = 300.0
    STOP_PATIENCE_COUNT = 10

    MAX_GLOBAL_STEP = 10000000
    LEARNING_RATE = 0.001

    GAMMA = 0.99
    BATCH_SIZE = 64

    AVG_EPISODE_SIZE_FOR_STAT = 50
    N_STEP = 1

    CLIP_GRAD = 0.2
