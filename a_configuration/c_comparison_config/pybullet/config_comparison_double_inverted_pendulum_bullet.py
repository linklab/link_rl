from a_configuration.a_base_config.config_comparison_base import ConfigComparisonBase
from a_configuration.b_single_config.open_ai_gym.config_cart_pole import ConfigCartPoleDqn, ConfigCartPoleReinforce, \
    ConfigCartPoleA2c
from a_configuration.b_single_config.pybullet.config_cart_pole_bullet import ConfigCartPoleBulletA2c
from a_configuration.b_single_config.pybullet.config_double_inverted_pendulum_bullet import \
    ConfigDoubleInvertedPendulumBulletSac


class ConfigComparisonDoubleInvertedPendulumBulletSac(ConfigComparisonBase):
    def __init__(self):
        ConfigComparisonBase.__init__(self)

        self.ENV_NAME = "InvertedDoublePendulumBulletEnv-v0"

        self.AGENT_PARAMETERS = [
            ConfigDoubleInvertedPendulumBulletSac(),
            ConfigDoubleInvertedPendulumBulletSac(),
            ConfigDoubleInvertedPendulumBulletSac(),
            ConfigDoubleInvertedPendulumBulletSac(),
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
