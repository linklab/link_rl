from a_configuration.a_base_config.config_comparison_base import ConfigComparisonBase
from a_configuration.b_single_config.open_ai_gym.config_cart_pole import ConfigCartPoleDqn
from a_configuration.b_single_config.pybullet.config_cart_pole_bullet import ConfigCartPoleBulletA2c, \
    ConfigCartPoleBulletDqn, ConfigCartPoleBulletDoubleDqn, ConfigCartPoleBulletDuelingDqn, \
    ConfigCartPoleBulletDoubleDuelingDqn


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

