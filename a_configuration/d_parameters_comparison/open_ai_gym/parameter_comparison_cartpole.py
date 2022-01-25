from a_configuration.b_base.parameter_base_comparison import ParameterComparisonBase
from a_configuration.c_parameters.open_ai_gym.parameter_cartpole import ParameterCartPoleDqn, \
    ParameterCartPoleReinforce, \
    ParameterCartPoleA2c, ParameterCartPoleDoubleDqn, ParameterCartPoleDuelingDqn, ParameterCartPoleDoubleDuelingDqn


class ParameterComparisonCartPoleDqn(ParameterComparisonBase):
    def __init__(self):
        ParameterComparisonBase.__init__(self)

        self.ENV_NAME = "CartPole-v1"

        self.AGENT_PARAMETERS = [
            ParameterCartPoleDqn(),
            ParameterCartPoleDqn(),
            ParameterCartPoleDqn()
        ]

        self.AGENT_PARAMETERS[0].N_STEP = 1
        self.AGENT_PARAMETERS[1].N_STEP = 2
        self.AGENT_PARAMETERS[2].N_STEP = 4
        self.AGENT_LABELS = [
            "DQN (N_STEP=1)",
            "DQN (N_STEP=2)",
            "DQN (N_STEP=4)",
        ]
        self.MAX_TRAINING_STEPS = 50_000
        self.N_RUNS = 5


class ParameterComparisonCartPoleDqnTypes(ParameterComparisonBase):
    def __init__(self):
        ParameterComparisonBase.__init__(self)

        self.ENV_NAME = "CartPole-v1"

        self.AGENT_PARAMETERS = [
            ParameterCartPoleDqn(),
            ParameterCartPoleDoubleDqn(),
            ParameterCartPoleDuelingDqn(),
            ParameterCartPoleDoubleDuelingDqn()
        ]

        self.AGENT_LABELS = [
            "DQN",
            "Double DQN",
            "Dueling DQN",
            "Double Dueling DQN",
        ]
        self.MAX_TRAINING_STEPS = 50_000
        self.N_RUNS = 5

# OnPolicy
class ParameterComparisonCartPoleReinforce(ParameterComparisonBase):
    def __init__(self):
        ParameterComparisonBase.__init__(self)

        self.AGENT_PARAMETERS = [
            ParameterCartPoleReinforce(),
            ParameterCartPoleReinforce(),
            ParameterCartPoleReinforce()
        ]


class ParameterComparisonCartPoleA2c(ParameterComparisonBase):
    def __init__(self):
        ParameterComparisonBase.__init__(self)

        self.AGENT_PARAMETERS = [
            ParameterCartPoleA2c(),
            ParameterCartPoleA2c(),
            ParameterCartPoleA2c(),
        ]
