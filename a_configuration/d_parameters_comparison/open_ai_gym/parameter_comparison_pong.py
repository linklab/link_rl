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

        self.AGENT_PARAMETERS[0].N_STEP = 1
        self.AGENT_PARAMETERS[1].N_STEP = 2
        self.AGENT_PARAMETERS[2].N_STEP = 3
        self.AGENT_LABELS = [
            "DQN (N_STEP=1)",
            "DQN (N_STEP=2)",
            "DQN (N_STEP=3)",
        ]
        self.MAX_TRAINING_STEPS = 1_000
        self.N_RUNS = 5


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