from codes.a_config._rl_parameters.on_policy.parameter_a2c import PARAMETERS_A2C
from codes.e_utils.names import *
from codes.a_config.parameters_general import PARAMETERS_GENERAL


class PARAMETERS_PONG_A2C(PARAMETERS_GENERAL, PARAMETERS_A2C):
    ENVIRONMENT_ID = EnvironmentName.PONG_NO_FRAME_SKIP_V4
    DEEP_LEARNING_MODEL = DeepLearningModelName.STOCHASTIC_DISCRETE_ACTOR_CRITIC_CNN
    RL_ALGORITHM = RLAlgorithmName.DISCRETE_A2C_V0
    OPTIMIZER = OptimizerName.ADAM
    # OPTIMIZER = OptimizerName.RMSProp

    TRAIN_STOP_EPISODE_REWARD = 20.0
    STOP_PATIENCE_COUNT = 10

    MAX_GLOBAL_STEP = 30000000
    ACTOR_LEARNING_RATE = 0.0001
    LEARNING_RATE = 0.0005

    GAMMA = 0.99
    BATCH_SIZE = 128

    AVG_EPISODE_SIZE_FOR_STAT = 50
    N_STEP = 4
    TRAIN_STEP_FREQ = 1

    CLIP_GRAD = 3.0
