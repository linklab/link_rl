from codes.a_config._rl_parameters.off_policy.parameter_ddpg import PARAMETERS_DDPG
from codes.a_config._rl_parameters.off_policy.parameter_td3 import PARAMETERS_TD3
from codes.a_config.parameters_general import PARAMETERS_GENERAL
from codes.e_utils.names import EnvironmentName, DeepLearningModelName, RLAlgorithmName, OptimizerName


class PARAMETERS_PENDULUM_TD3(PARAMETERS_GENERAL, PARAMETERS_TD3):
    ENVIRONMENT_ID = EnvironmentName.PENDULUM_V0
    DEEP_LEARNING_MODEL = DeepLearningModelName.TD3_MLP
    RL_ALGORITHM = RLAlgorithmName.TD3_V0
    OPTIMIZER = OptimizerName.ADAM

    STOP_MEAN_EPISODE_REWARD = -140
    STOP_PATIENCE_COUNT = 10
    REPLAY_BUFFER_SIZE = 100000
    MAX_GLOBAL_STEP = 250000
    LEARNING_RATE = 0.001
    ACTOR_LEARNING_RATE = 0.001
    TRAIN_STEP_FREQ = 1
    GAMMA = 0.99
    BATCH_SIZE = 128
    AVG_STEP_SIZE_FOR_TRAIN_LOSS = 50
    AVG_EPISODE_SIZE_FOR_STAT = 50
    N_STEP = 4
    OMEGA = False

    EPSILON_INIT = 0.9
    EPSILON_MIN = 0.001
    EPSILON_MIN_STEP = 50000

    CLIP_GRAD = 0.1

    ACTION_SCALE = 2.0

    ACT_NOISE = 0.1
    NOISE_CLIP = 0.5
