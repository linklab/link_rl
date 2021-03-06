from codes.a_config._rl_parameters.on_policy.parameter_on_policy import PARAMETERS_ON_POLICY
from codes.a_config.parameters_general import StochasticActionType, StochasticActionSelectorType


class PARAMETERS_PPO(PARAMETERS_ON_POLICY):
    PPO_TRAJECTORY_SIZE = 2049
    PPO_TRAJECTORY_BATCH_SIZE = 64
    PPO_EPSILON_CLIP = 0.2
    PPO_GAE_LAMBDA = 0.70
    PPO_K_EPOCHS = 10
    ACTOR_LEARNING_RATE = 0.0002
    LEARNING_RATE = 0.001

    TRAIN_STEP_FREQ = 1
    N_STEP = 1

    TYPE_OF_STOCHASTIC_ACTION = StochasticActionType.SAMPLE
    TYPE_OF_STOCHASTIC_ACTION_SELECTOR = StochasticActionSelectorType.BASIC_ACTION_SELECTOR