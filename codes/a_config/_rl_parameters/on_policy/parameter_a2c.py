from codes.a_config._rl_parameters.on_policy.parameter_on_policy import PARAMETERS_ON_POLICY
from codes.a_config.parameters_general import StochasticActionType, StochasticActionSelectorType


class PARAMETERS_A2C(PARAMETERS_ON_POLICY):
    ACTOR_LEARNING_RATE = 0.0001
    LEARNING_RATE = 0.001

    TYPE_OF_STOCHASTIC_ACTION = StochasticActionType.SAMPLE
    TYPE_OF_STOCHASTIC_ACTION_SELECTOR = StochasticActionSelectorType.BASIC_ACTION_SELECTOR