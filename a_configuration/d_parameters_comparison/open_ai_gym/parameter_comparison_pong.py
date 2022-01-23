from a_configuration.b_base.parameter_base_comparison import ParameterComparisonBase
from a_configuration.c_parameters.open_ai_gym.parameter_pong import ParameterPongDqn, ParameterPongDoubleDqn, \
    ParameterPongDuelingDqn, ParameterPongDoubleDuelingDqn


class ParameterComparisonPongDqn(ParameterComparisonBase):
    def __init__(self):
        ParameterComparisonBase.__init__(self)
        self.ENV_NAME = "PongNoFrameskip-v4"

        self.MAX_TRAINING_STEPS = 1_000_000

        self.AGENT_PARAMETERS = [
            ParameterPongDqn(),
            ParameterPongDqn(),
            ParameterPongDqn()
        ]


class ParameterComparisonPongDqnTypes(ParameterComparisonBase):
    def __init__(self):
        ParameterComparisonBase.__init__(self)

        self.ENV_NAME = "PongNoFrameskip-v4"

        self.MAX_TRAINING_STEPS = 1_000_000

        self.AGENT_PARAMETERS = [
            ParameterPongDqn(),
            ParameterPongDoubleDqn(),
            ParameterPongDuelingDqn(),
            ParameterPongDoubleDuelingDqn()
        ]