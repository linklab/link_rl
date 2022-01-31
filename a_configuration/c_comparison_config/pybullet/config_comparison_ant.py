from a_configuration.a_base_config.config_comparison_base import ConfigComparisonBase
from a_configuration.b_single_config.pybullet.config_ant_bullet import ConfigAntBulletSac, ConfigAntBulletDdpg, \
    ConfigAntBulletTd3, ConfigAntBulletPpoTrajectory


class ConfigComparisonAntBulletDDpgTd3(ConfigComparisonBase):
    def __init__(self):
        ConfigComparisonBase.__init__(self)

        self.ENV_NAME = "AntBulletEnv-v0"

        self.AGENT_PARAMETERS = [
            ConfigAntBulletDdpg(),
            ConfigAntBulletTd3(),
        ]

        self.AGENT_LABELS = [
            "DDPG",
            "TD3",
        ]
        self.MAX_TRAINING_STEPS = 300_000
        self.N_RUNS = 5


class ConfigComparisonAntBulletSac(ConfigComparisonBase):
    def __init__(self):
        ConfigComparisonBase.__init__(self)

        self.ENV_NAME = "AntBulletEnv-v0"

        self.AGENT_PARAMETERS = [
            ConfigAntBulletSac(),
            ConfigAntBulletSac(),
            ConfigAntBulletSac(),
            ConfigAntBulletSac(),
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
        self.MAX_TRAINING_STEPS = 500_000
        self.N_RUNS = 5


class ConfigComparisonAntBulletPpoSac(ConfigComparisonBase):
    def __init__(self):
        ConfigComparisonBase.__init__(self)

        self.ENV_NAME = "AntBulletEnv-v0"

        self.AGENT_PARAMETERS = [
            ConfigAntBulletPpoTrajectory(),
            ConfigAntBulletSac(),
        ]

        self.AGENT_LABELS = [
            "ppo_trajectory",
            "sac",
        ]
        self.MAX_TRAINING_STEPS = 500_000
        self.N_RUNS = 5
