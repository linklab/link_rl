from a_configuration.a_base_config.config_comparison_base import ConfigComparisonBase
from a_configuration.b_single_config.open_ai_gym.config_cart_pole import ConfigCartPoleDqn, ConfigCartPoleReinforce, \
    ConfigCartPoleA2c
from a_configuration.b_single_config.pybullet.config_cart_pole_bullet import ConfigCartPoleBulletA2c, \
    ConfigCartPoleBulletDqn, ConfigCartPoleBulletDoubleDqn, ConfigCartPoleBulletDuelingDqn, \
    ConfigCartPoleBulletDoubleDuelingDqn
from a_configuration.b_single_config.pybullet.config_cart_pole_continuous_bullet import \
    ConfigCartPoleContinuousBulletDdpg
from g_utils.types import ModelType


class ConfigComparisonCartPoleBulletDqn(ConfigComparisonBase):
    def __init__(self):
        ConfigComparisonBase.__init__(self)

        self.ENV_NAME = "CartPoleBulletEnv-v1"

        self.AGENT_PARAMETERS = [
            ConfigCartPoleDqn(),
            ConfigCartPoleDqn(),
            ConfigCartPoleDqn()
        ]


class ConfigComparisonCartPoleBulletDqnTypes(ConfigComparisonBase):
    def __init__(self):
        ConfigComparisonBase.__init__(self)

        self.ENV_NAME = "CartPoleBulletEnv-v1"

        self.AGENT_PARAMETERS = [
            ConfigCartPoleBulletDqn(),
            ConfigCartPoleBulletDoubleDqn(),
            ConfigCartPoleBulletDuelingDqn(),
            ConfigCartPoleBulletDoubleDuelingDqn()
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
class ConfigComparisonCartPoleBulletA2c(ConfigComparisonBase):
    def __init__(self):
        ConfigComparisonBase.__init__(self)

        self.ENV_NAME = "CartPoleBulletEnv-v1"

        self.AGENT_PARAMETERS = [
            ConfigCartPoleBulletA2c(),
            ConfigCartPoleBulletA2c(),
            ConfigCartPoleBulletA2c(),
        ]

        self.AGENT_PARAMETERS[0].LEARNING_RATE = 0.001
        self.AGENT_PARAMETERS[1].LEARNING_RATE = 0.0001
        self.AGENT_PARAMETERS[2].LEARNING_RATE = 0.00001
        self.AGENT_LABELS = [
            "DQN (LEARNING_RATE = 0.001)",
            "DQN (LEARNING_RATE = 0.0001)",
            "DQN (LEARNING_RATE = 0.00001)",
        ]
        self.MAX_TRAINING_STEPS = 50_000
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
