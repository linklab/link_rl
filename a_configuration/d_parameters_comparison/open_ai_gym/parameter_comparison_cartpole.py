from a_configuration.b_base.parameter_base_comparison import ParameterComparisonBase
from a_configuration.c_parameters.open_ai_gym.parameter_cartpole import ParameterCartPoleDqn, ParameterCartPoleReinforce, \
    ParameterCartPoleA2c


class ParameterComparisonCartPoleDqn(ParameterComparisonBase):
    def __init__(self):
        ParameterComparisonBase.__init__(self)

        self.ENV_NAME = "CartPole-v1"

        self.AGENT_PARAMETERS = [
            ParameterCartPoleDqn(),
            ParameterCartPoleDqn(),
            ParameterCartPoleDqn()
        ]


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
