from codes.a_config._rl_parameters.off_policy.parameter_ddpg import PARAMETERS_DDPG, DDPGActionType, \
    DDPGActionSelectorType
from codes.a_config._rl_parameters.off_policy.parameter_sac import PARAMETERS_SAC, SACActionType, SACActionSelectorType
from codes.a_config.parameters_general import PARAMETERS_GENERAL
from codes.e_utils.names import EnvironmentName, DeepLearningModelName, RLAlgorithmName, OptimizerName


class PARAMETERS_PENDULUM_SAC(PARAMETERS_GENERAL, PARAMETERS_SAC):
    ENVIRONMENT_ID = EnvironmentName.PENDULUM_V0
    DEEP_LEARNING_MODEL = DeepLearningModelName.SOFT_ACTOR_CRITIC_MLP
    RL_ALGORITHM = RLAlgorithmName.SAC_V0
    OPTIMIZER = OptimizerName.ADAM

    TRAIN_STOP_EPISODE_REWARD = -140
    TRAIN_STOP_EPISODE_REWARD_STD = 10

    STOP_PATIENCE_COUNT = 10

    REPLAY_BUFFER_SIZE = 100000
    LEARNING_RATE = 0.001
    ACTOR_LEARNING_RATE = 0.0002
    TRAIN_STEP_FREQ = 1
    GAMMA = 0.99

    BATCH_SIZE = 128
    MIN_REPLAY_SIZE_FOR_TRAIN = 2048

    AVG_STEP_SIZE_FOR_TRAIN_LOSS = 50
    AVG_EPISODE_SIZE_FOR_STAT = 50
    N_STEP = 1
    OMEGA = False

    CLIP_GRAD = 0.1

    ACTION_SCALE = 2.0

    NOISE_ENABLED = True

    EPSILON_INIT = 1.0
    EPSILON_MIN = 0.01
    EPSILON_MIN_STEP = 20000
    MAX_GLOBAL_STEP = 1000000

    TRAIN_ONLY_AFTER_EPISODE = False
    NUM_TRAIN_ONLY_AFTER_EPISODE = 100

    TYPE_OF_SAC_ACTION = SACActionType.GAUSSIAN_NOISE_WITH_EPSILON
    TYPE_OF_SAC_ACTION_SELECTOR = SACActionSelectorType.BASIC_ACTION_SELECTOR

