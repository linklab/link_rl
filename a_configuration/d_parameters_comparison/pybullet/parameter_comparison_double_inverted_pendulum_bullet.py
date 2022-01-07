from a_configuration.b_base.parameter_base_comparison import ParameterComparisonBase
from a_configuration.c_parameters.open_ai_gym.parameter_cartpole import ParameterCartPoleDqn, ParameterCartPoleReinforce, \
    ParameterCartPoleA2c
from a_configuration.c_parameters.pybullet.parameter_cartpole_bullet import ParameterCartPoleBulletA2c
from a_configuration.c_parameters.pybullet.parameter_double_inverted_pendulum_bullet import \
    ParameterDoubleInvertedPendulumBulletSac


class ParameterComparisonDoubleInvertedPendulumBulletSac(ParameterComparisonBase):
    def __init__(self):
        ParameterComparisonBase.__init__(self)

        self.ENV_NAME = "InvertedDoublePendulumBulletEnv-v0"

        self.AGENT_PARAMETERS = [
            ParameterDoubleInvertedPendulumBulletSac(),
            ParameterDoubleInvertedPendulumBulletSac(),
            ParameterDoubleInvertedPendulumBulletSac(),
            ParameterDoubleInvertedPendulumBulletSac(),
        ]
