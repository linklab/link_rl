from a_configuration.b_base.parameter_base_comparison import ParameterComparisonBase
from a_configuration.c_parameters.open_ai_gym.parameter_cartpole import ParameterCartPoleDqn, ParameterCartPoleReinforce, \
    ParameterCartPoleA2c
from a_configuration.c_parameters.pybullet.parameter_cartpole_bullet import ParameterCartPoleBulletA2c, \
    ParameterCartPoleBulletDqn, ParameterCartPoleBulletDoubleDqn, ParameterCartPoleBulletDuelingDqn, \
    ParameterCartPoleBulletDoubleDuelingDqn


class ParameterComparisonCartPoleBulletDqn(ParameterComparisonBase):
    def __init__(self):
        ParameterComparisonBase.__init__(self)

        self.ENV_NAME = "CartPole-v1"

        self.AGENT_PARAMETERS = [
            ParameterCartPoleDqn(),
            ParameterCartPoleDqn(),
            ParameterCartPoleDqn()
        ]


class ParameterComparisonCartPoleBulletDqnTypes(ParameterComparisonBase):
    def __init__(self):
        ParameterComparisonBase.__init__(self)

        self.ENV_NAME = "CartPole-v1"

        self.AGENT_PARAMETERS = [
            ParameterCartPoleBulletDqn(),
            ParameterCartPoleBulletDoubleDqn(),
            ParameterCartPoleBulletDuelingDqn(),
            ParameterCartPoleBulletDoubleDuelingDqn()
        ]


# OnPolicy
class ParameterComparisonCartPoleBulletA2c(ParameterComparisonBase):
    def __init__(self):
        ParameterComparisonBase.__init__(self)

        self.ENV_NAME = "CartPoleBulletEnv-v1"

        self.AGENT_PARAMETERS = [
            ParameterCartPoleBulletA2c(),
            ParameterCartPoleBulletA2c(),
            ParameterCartPoleBulletA2c(),
        ]
