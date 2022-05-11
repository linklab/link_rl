from a_configuration.a_base_config.config_comparison_base import ConfigComparisonBase
from a_configuration.b_single_config.pybullet.config_cart_pole_continuous_bullet import \
    ConfigCartPoleContinuousBulletDdpg, ConfigCartPoleContinuousBulletTd3, ConfigCartPoleContinuousBulletSac, \
    ConfigCartPoleContinuousBulletA2c, ConfigCartPoleContinuousBulletPpoTrajectory, ConfigCartPoleContinuousBulletPpo
from g_utils.types import ModelType


class ConfigComparisonCartPoleContinuousBulletDdpgTd3(ConfigComparisonBase):
    def __init__(self):
        ConfigComparisonBase.__init__(self)

        self.ENV_NAME = "CartPoleContinuousBulletEnv-v0"

        self.AGENT_PARAMETERS = [
            ConfigCartPoleContinuousBulletDdpg(),
            ConfigCartPoleContinuousBulletTd3(),
        ]

        self.AGENT_LABELS = [
            "DDPG",
            "TD3",
        ]

        self.MAX_TRAINING_STEPS = 10_000
        self.N_RUNS = 5


class ConfigComparisonCartPoleContinuousBulletDdpg(ConfigComparisonBase):
    def __init__(self):
        ConfigComparisonBase.__init__(self)

        self.ENV_NAME = "CartPoleContinuousBulletEnv-v0"

        self.AGENT_PARAMETERS = [
            ConfigCartPoleContinuousBulletDdpg(),
            ConfigCartPoleContinuousBulletDdpg(),
            ConfigCartPoleContinuousBulletDdpg(),
        ]

        self.AGENT_PARAMETERS[0].MODEL_TYPE = ModelType.SMALL_RECURRENT
        self.AGENT_PARAMETERS[1].MODEL_TYPE = ModelType.SMALL_LINEAR
        self.AGENT_PARAMETERS[2].MODEL_TYPE = ModelType.SMALL_LINEAR_2
        self.AGENT_LABELS = [
            "DDPG + GRU",
            "DDPG + Linear",
            "DDPG + Linear_2",
        ]
        self.MAX_TRAINING_STEPS = 50_000
        self.N_RUNS = 5


class ConfigComparisonCartPoleContinuousBulletAll(ConfigComparisonBase):
    def __init__(self):
        ConfigComparisonBase.__init__(self)

        self.ENV_NAME = "CartPoleContinuousBulletEnv-v0"

        self.AGENT_PARAMETERS = [
            ConfigCartPoleContinuousBulletDdpg(),
            ConfigCartPoleContinuousBulletTd3(),
            ConfigCartPoleContinuousBulletSac(),
            ConfigCartPoleContinuousBulletA2c(),
            ConfigCartPoleContinuousBulletPpoTrajectory(),
        ]

        self.AGENT_LABELS = [
            "ddpg",
            "td3",
            "sac",
            "a2c",
            "ppo_trajectory"
        ]
        self.MAX_TRAINING_STEPS = 100_000
        self.N_RUNS = 5


class ConfigComparisonCartPoleContinuousBulletPpo(ConfigComparisonBase):
    def __init__(self):
        ConfigComparisonBase.__init__(self)

        self.ENV_NAME = "CartPoleContinuousBulletEnv-v0"

        self.AGENT_PARAMETERS = [
            ConfigCartPoleContinuousBulletPpo(),
            ConfigCartPoleContinuousBulletPpoTrajectory(),
        ]

        self.AGENT_LABELS = [
            "PPO",
            "PPO Trajectory"
        ]
        self.MAX_TRAINING_STEPS = 10_000
        self.N_RUNS = 5
