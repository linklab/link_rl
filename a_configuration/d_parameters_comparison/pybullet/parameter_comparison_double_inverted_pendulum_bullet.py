from a_configuration.b_base.parameter_base_comparison import ParameterComparisonBase
from a_configuration.c_parameters.open_ai_gym.parameter_cart_pole import ParameterCartPoleDqn, ParameterCartPoleReinforce, \
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

        self.AGENT_PARAMETERS[0].AUTOMATIC_ENTROPY_TEMPERATURE_TUNING = False
        self.AGENT_PARAMETERS[0].DEFAULT_ALPHA = 0.2
        self.AGENT_PARAMETERS[1].AUTOMATIC_ENTROPY_TEMPERATURE_TUNING = False
        self.AGENT_PARAMETERS[1].DEFAULT_ALPHA = 0.5
        self.AGENT_PARAMETERS[2].AUTOMATIC_ENTROPY_TEMPERATURE_TUNING = True
        self.AGENT_PARAMETERS[2].MIN_ALPHA = 0.0
        self.AGENT_PARAMETERS[3].AUTOMATIC_ENTROPY_TEMPERATURE_TUNING = True
        self.AGENT_PARAMETERS[3].MIN_ALPHA = 0.2
        self.AGENT_LABELS = [
            "alpha = 0.2",
            "alpha = 0.5",
            "alpha tuning (No Alpha Limit)",
            "alpha tuning (Min Alpha = 0.2)",
        ]
        self.MAX_TRAINING_STEPS = 100000
        self.N_RUNS = 5
