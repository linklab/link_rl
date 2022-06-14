from link_rl.a_configuration.a_base_config.b_agents.config_agents_off_policy import ConfigDqn, ConfigDoubleDqn, ConfigDuelingDqn, \
    ConfigDoubleDuelingDqn
from link_rl.a_configuration.a_base_config.a_environments.open_ai_gym.config_gym_atari import ConfigBreakout
from link_rl.a_configuration.a_base_config.b_agents.config_agents_on_policy import ConfigA2c, ConfigA3c, ConfigPpo, ConfigPpoTrajectory
from link_rl.a_configuration.a_base_config.config_single_base import ConfigBase
from link_rl.g_utils.types import ModelType


class ConfigBreakoutDqn(ConfigBase, ConfigBreakout, ConfigDqn):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigBreakout.__init__(self)
        ConfigDqn.__init__(self)

        self.MAX_TRAINING_STEPS = 1_000_000
        self.BUFFER_CAPACITY = 500_000
        self.MODEL_TYPE = ModelType.MEDIUM_2D_CONVOLUTIONAL


class ConfigBreakoutDoubleDqn(ConfigBase, ConfigBreakout, ConfigDoubleDqn):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigBreakout.__init__(self)
        ConfigDoubleDqn.__init__(self)

        self.MAX_TRAINING_STEPS = 1_000_000
        self.BUFFER_CAPACITY = 500_000
        self.MODEL_TYPE = ModelType.MEDIUM_2D_CONVOLUTIONAL


class ConfigBreakoutDuelingDqn(ConfigBase, ConfigBreakout, ConfigDuelingDqn):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigBreakout.__init__(self)
        ConfigDuelingDqn.__init__(self)

        self.MAX_TRAINING_STEPS = 1_000_000
        self.BUFFER_CAPACITY = 500_000
        self.MODEL_TYPE = ModelType.MEDIUM_2D_CONVOLUTIONAL


class ConfigBreakoutDoubleDuelingDqn(ConfigBase, ConfigBreakout, ConfigDoubleDuelingDqn):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigBreakout.__init__(self)
        ConfigDoubleDuelingDqn.__init__(self)

        self.MAX_TRAINING_STEPS = 1_000_000
        self.BUFFER_CAPACITY = 500_000
        self.MODEL_TYPE = ModelType.MEDIUM_2D_CONVOLUTIONAL


class ConfigBreakoutA2c(ConfigBase, ConfigBreakout, ConfigA2c):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigBreakout.__init__(self)
        ConfigA2c.__init__(self)

        self.MAX_TRAINING_STEPS = 1_000_000
        self.MODEL_TYPE = ModelType.MEDIUM_2D_CONVOLUTIONAL


class ConfigBreakoutA3c(ConfigBase, ConfigBreakout, ConfigA3c):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigBreakout.__init__(self)
        ConfigA3c.__init__(self)

        self.MAX_TRAINING_STEPS = 1_000_000
        self.MODEL_TYPE = ModelType.MEDIUM_2D_CONVOLUTIONAL


class ConfigBreakoutPpo(ConfigBase, ConfigBreakout, ConfigPpo):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigBreakout.__init__(self)
        ConfigPpo.__init__(self)

        self.MAX_TRAINING_STEPS = 1_000_000
        self.MODEL_TYPE = ModelType.MEDIUM_2D_CONVOLUTIONAL


class ConfigBreakoutPpoTrajectory(ConfigBase, ConfigBreakout, ConfigPpoTrajectory):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigBreakout.__init__(self)
        ConfigPpoTrajectory.__init__(self)

        self.MAX_TRAINING_STEPS = 1_000_000
        self.MODEL_TYPE = ModelType.MEDIUM_2D_CONVOLUTIONAL
