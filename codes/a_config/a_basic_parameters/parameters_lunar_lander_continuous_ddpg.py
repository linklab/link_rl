# https://towardsdatascience.com/solving-lunar-lander-openaigym-reinforcement-learning-785675066197
from codes.a_config._rl_parameters.off_policy.parameter_ddpg import PARAMETERS_DDPG
from codes.e_utils.names import *
from codes.a_config.parameters_general import PARAMETERS_GENERAL


class PARAMETERS_LUNAR_LANDER_CONTINUOUS_DDPG(PARAMETERS_GENERAL, PARAMETERS_DDPG):
    ENVIRONMENT_ID = EnvironmentName.LUNAR_LANDER_CONTINUOUS_V2
    DEEP_LEARNING_MODEL = DeepLearningModelName.DETERMINISTIC_CONTINUOUS_ACTOR_CRITIC_MLP
    RL_ALGORITHM = RLAlgorithmName.DDPG_V0
    OPTIMIZER = OptimizerName.ADAM

    STOP_MEAN_EPISODE_REWARD = 180.0
    STOP_PATIENCE_COUNT = 10

    REPLAY_BUFFER_SIZE = 100000
    MAX_GLOBAL_STEP = 1000000
    LEARNING_RATE = 0.001
    ACTOR_LEARNING_RATE = 0.0001
    TRAIN_STEP_FREQ = 1
    GAMMA = 0.99
    BATCH_SIZE = 64
    AVG_STEP_SIZE_FOR_TRAIN_LOSS = 50
    AVG_EPISODE_SIZE_FOR_STAT = 50
    N_STEP = 4
    OMEGA = False

    EPSILON_INIT = 0.9
    EPSILON_MIN = 0.001
    EPSILON_MIN_STEP = 200000

    CLIP_GRAD = 0.1

    ACTION_SCALE = 1.0
