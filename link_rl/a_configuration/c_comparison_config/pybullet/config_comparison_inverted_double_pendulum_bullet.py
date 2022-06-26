from link_rl.a_configuration.a_base_config.config_comparison_base import ConfigComparisonBase
from link_rl.a_configuration.b_single_config.pybullet.config_inverted_double_pendulum_bullet import \
    ConfigInvertedDoublePendulumBulletSac, ConfigInvertedDoublePendulumBulletA2c, ConfigInvertedDoublePendulumBulletPpo, \
    ConfigInvertedDoublePendulumBulletPpoTrajectory


class ConfigComparisonInvertedDoublePendulumBulletA2c(ConfigComparisonBase):
    def __init__(self):
        ConfigComparisonBase.__init__(self)

        self.ENV_NAME = "InvertedDoublePendulumBulletEnv-v0"

        self.AGENT_PARAMETERS = [
            ConfigInvertedDoublePendulumBulletA2c(),
            ConfigInvertedDoublePendulumBulletA2c(),
            ConfigInvertedDoublePendulumBulletA2c(),
        ]
        self.AGENT_PARAMETERS[0].USE_GAE = False
        self.AGENT_PARAMETERS[0].USE_BOOTSTRAP_FOR_TARGET_VALUE = False

        self.AGENT_PARAMETERS[1].USE_GAE = False
        self.AGENT_PARAMETERS[1].USE_BOOTSTRAP_FOR_TARGET_VALUE = True

        self.AGENT_PARAMETERS[2].USE_GAE = True

        self.AGENT_LABELS = [
            "a2c + no gae + full return",
            "a2c + no gae + td bootstrap",
            "a2c + gae",
        ]
        self.MAX_TRAINING_STEPS = 300_000
        self.N_RUNS = 5


class ConfigComparisonInvertedDoublePendulumBulletSacAlpha(ConfigComparisonBase):
    def __init__(self):
        ConfigComparisonBase.__init__(self)

        self.ENV_NAME = "InvertedDoublePendulumBulletEnv-v0"

        self.AGENT_PARAMETERS = [
            ConfigInvertedDoublePendulumBulletSac(),
            ConfigInvertedDoublePendulumBulletSac(),
            ConfigInvertedDoublePendulumBulletSac(),
            ConfigInvertedDoublePendulumBulletSac(),
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
        self.MAX_TRAINING_STEPS = 100_000
        self.N_RUNS = 5


class ConfigComparisonInvertedDoublePendulumBulletSacPer(ConfigComparisonBase):
    def __init__(self):
        ConfigComparisonBase.__init__(self)

        self.ENV_NAME = "InvertedDoublePendulumBulletEnv-v0"

        self.AGENT_PARAMETERS = [
            ConfigInvertedDoublePendulumBulletSac(),
            ConfigInvertedDoublePendulumBulletSac()
        ]

        self.AGENT_PARAMETERS[1].USE_PER = True
        self.AGENT_LABELS = [
            "sac",
            "sac + per"
        ]
        self.MAX_TRAINING_STEPS = 100_000
        self.N_RUNS = 5


class ConfigComparisonInvertedDoublePendulumBulletA2cPpo(ConfigComparisonBase):
    def __init__(self):
        ConfigComparisonBase.__init__(self)

        self.ENV_NAME = "InvertedDoublePendulumBulletEnv-v0"

        self.AGENT_PARAMETERS = [
            ConfigInvertedDoublePendulumBulletA2c(),
            ConfigInvertedDoublePendulumBulletPpo(),
            ConfigInvertedDoublePendulumBulletPpoTrajectory(),
        ]

        self.AGENT_LABELS = [
            "a2c",
            "ppo",
            "ppo_trajectory",
        ]
        self.MAX_TRAINING_STEPS = 300_000
        self.N_RUNS = 5


class ConfigComparisonInvertedDoublePendulumBulletSacPpo(ConfigComparisonBase):
    def __init__(self):
        ConfigComparisonBase.__init__(self)

        self.ENV_NAME = "InvertedDoublePendulumBulletEnv-v0"

        self.AGENT_PARAMETERS = [
            ConfigInvertedDoublePendulumBulletSac(),
            ConfigInvertedDoublePendulumBulletPpo()
        ]

        self.AGENT_LABELS = [
            "sac",
            "ppo"
        ]
        self.MAX_TRAINING_STEPS = 300_000
        self.N_RUNS = 5