from codes.a_config._rl_parameters.on_policy.parameter_a2c import PARAMETERS_A2C
from codes.e_utils.names import *
from codes.a_config.parameters_general import PARAMETERS_GENERAL


class PARAMETERS_CARTPOLE_A2C(PARAMETERS_GENERAL, PARAMETERS_A2C):
    ENVIRONMENT_ID = EnvironmentName.CARTPOLE_V1
    DEEP_LEARNING_MODEL = DeepLearningModelName.DISCRETE_STOCHASTIC_ACTOR_CRITIC_MLP
    #DEEP_LEARNING_MODEL = DeepLearningModelName.DISCRETE_STOCHASTIC_ACTOR_CRITIC_RNN
    RL_ALGORITHM = RLAlgorithmName.DISCRETE_A2C_V0
    OPTIMIZER = OptimizerName.ADAM

    TRAIN_STOP_EPISODE_REWARD = 490.0
    STOP_PATIENCE_COUNT = 10

    MAX_GLOBAL_STEP = 2000000
    ACTOR_LEARNING_RATE = 0.0001
    LEARNING_RATE = 0.001

    GAMMA = 0.99
    BATCH_SIZE = 32

    AVG_EPISODE_SIZE_FOR_STAT = 50
    N_STEP = 1

    CLIP_GRAD = 3.0
