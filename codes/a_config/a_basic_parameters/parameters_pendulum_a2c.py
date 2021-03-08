from codes.a_config._rl_parameters.on_policy.parameter_a2c import PARAMETERS_A2C
from codes.e_utils.names import *
from codes.a_config.parameters_general import PARAMETERS_GENERAL


class PARAMETERS_PENDULUM_A2C(PARAMETERS_GENERAL, PARAMETERS_A2C):
    ENVIRONMENT_ID = EnvironmentName.PENDULUM_V0
    DEEP_LEARNING_MODEL = DeepLearningModelName.STOCHASTIC_CONTINUOUS_ACTOR_CRITIC_MLP
    RL_ALGORITHM = RLAlgorithmName.CONTINUOUS_A2C_V0
    OPTIMIZER = OptimizerName.ADAM

    STOP_MEAN_EPISODE_REWARD = -140.0
    STOP_PATIENCE_COUNT = 10

    MAX_GLOBAL_STEP = 15000000
    LEARNING_RATE = 0.0005

    GAMMA = 0.99
    BATCH_SIZE = 64

    AVG_EPISODE_SIZE_FOR_STAT = 50
    N_STEP = 1
    TRAIN_STEP_FREQ = 1

    CLIP_GRAD = 0.2

    ACTION_SCALE = 2.0
