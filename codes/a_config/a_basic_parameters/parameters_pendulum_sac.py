from codes.a_config._rl_parameters.off_policy.parameter_sac import PARAMETERS_SAC, StochasticActionType, \
    StochasticActionSelectorType
from codes.a_config.parameters_general import PARAMETERS_GENERAL
from codes.e_utils.names import EnvironmentName, DeepLearningModelName, RLAlgorithmName, OptimizerName


class PARAMETERS_PENDULUM_SAC(PARAMETERS_GENERAL, PARAMETERS_SAC):
    ENVIRONMENT_ID = EnvironmentName.PENDULUM_V0
    DEEP_LEARNING_MODEL = DeepLearningModelName.CONTINUOUS_SAC_MLP
    RL_ALGORITHM = RLAlgorithmName.CONTINUOUS_SAC_V0
    OPTIMIZER = OptimizerName.RMSProp

    TRAIN_STOP_EPISODE_REWARD = -140
    TRAIN_STOP_EPISODE_REWARD_STD = 10

    STOP_PATIENCE_COUNT = 10

    REPLAY_BUFFER_SIZE = 100000
    LEARNING_RATE = 0.001
    ACTOR_LEARNING_RATE = 0.0001
    GAMMA = 0.99
    BATCH_SIZE = 128

    AVG_STEP_SIZE_FOR_TRAIN_LOSS = 50
    AVG_EPISODE_SIZE_FOR_STAT = 50
    N_STEP = 1
    OMEGA = False

    CLIP_GRAD = 3.0

    TRAIN_ONLY_AFTER_EPISODE = False
    NUM_TRAIN_ONLY_AFTER_EPISODE = 100

    TYPE_OF_STOCHASTIC_ACTION = StochasticActionType.SAMPLE
    TYPE_OF_STOCHASTIC_ACTION_SELECTOR = StochasticActionSelectorType.BASIC_ACTION_SELECTOR

    TRAIN_STEP_FREQ = 2
    POLICY_UPDATE_FREQUENCY = 2 * TRAIN_STEP_FREQ

    VERBOSE_TO_LOG = False
