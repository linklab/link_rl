from codes.a_config._rl_parameters.off_policy.parameter_ddpg import PARAMETERS_DDPG
from codes.e_utils.names import *
from codes.a_config.parameters_general import PARAMETERS_GENERAL


class PARAMETERS_PENDULUM_D4PG(PARAMETERS_GENERAL, PARAMETERS_DDPG):
    ENVIRONMENT_ID = EnvironmentName.PENDULUM_V0
    DEEP_LEARNING_MODEL = DeepLearningModelName.DETERMINISTIC_CONTINUOUS_ACTOR_CRITIC_MLP
    RL_ALGORITHM = RLAlgorithmName.D4PG_V0

    TRAIN_STOP_EPISODE_REWARD = -200
    STOP_PATIENCE_COUNT = 10
    REPLAY_BUFFER_SIZE = 500000
    MAX_GLOBAL_STEP = 500000
    LEARNING_RATE = 0.001
    ACTOR_LEARNING_RATE = 0.0001
    TRAIN_STEP_FREQ = 1
    GAMMA = 0.99
    BATCH_SIZE = 64
    AVG_EPISODE_SIZE_FOR_STAT = 50
    N_STEP = 4
    OMEGA = False

    CLIP_GRAD = 3.0

    ACTION_SCALE = 2.0

    V_MAX = 0
    V_MIN = -1800
    N_ATOMS = 251
