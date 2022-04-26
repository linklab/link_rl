from a_configuration.a_base_config.b_agents.config_agents_off_policy import ConfigDqn, ConfigDoubleDqn, ConfigDuelingDqn, \
    ConfigDoubleDuelingDqn
from a_configuration.a_base_config.a_environments.open_ai_gym.config_gym_atari import ConfigVideoPinball
from a_configuration.a_base_config.b_agents.config_agents_on_policy import ConfigA2c, ConfigA3c, ConfigPpo, ConfigPpoTrajectory
from a_configuration.a_base_config.config_single_base import ConfigBase
from g_utils.types import ModelType


class ConfigVideoPinballDqn(ConfigBase, ConfigVideoPinball, ConfigDqn):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigVideoPinball.__init__(self)
        ConfigDqn.__init__(self)

        self.MAX_TRAINING_STEPS = 1_000_000
        self.BUFFER_CAPACITY = 500_000
        self.MODEL_TYPE = ModelType.MEDIUM_2D_CONVOLUTIONAL


class ConfigVideoPinballDoubleDqn(ConfigBase, ConfigVideoPinball, ConfigDoubleDqn):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigVideoPinball.__init__(self)
        ConfigDoubleDqn.__init__(self)

        self.MAX_TRAINING_STEPS = 1_000_000
        self.BUFFER_CAPACITY = 500_000
        self.MODEL_TYPE = ModelType.MEDIUM_2D_CONVOLUTIONAL


class ConfigVideoPinballDuelingDqn(ConfigBase, ConfigVideoPinball, ConfigDuelingDqn):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigVideoPinball.__init__(self)
        ConfigDuelingDqn.__init__(self)

        self.MAX_TRAINING_STEPS = 1_000_000
        self.BUFFER_CAPACITY = 500_000
        self.MODEL_TYPE = ModelType.MEDIUM_2D_CONVOLUTIONAL


class ConfigVideoPinballDoubleDuelingDqn(ConfigBase, ConfigVideoPinball, ConfigDoubleDuelingDqn):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigVideoPinball.__init__(self)
        ConfigDoubleDuelingDqn.__init__(self)

        self.MAX_TRAINING_STEPS = 1_000_000
        self.BUFFER_CAPACITY = 500_000
        self.MODEL_TYPE = ModelType.MEDIUM_2D_CONVOLUTIONAL


class ConfigVideoPinballA2c(ConfigBase, ConfigVideoPinball, ConfigA2c):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigVideoPinball.__init__(self)
        ConfigA2c.__init__(self)

        self.MAX_TRAINING_STEPS = 1_000_000
        self.MODEL_TYPE = ModelType.MEDIUM_2D_CONVOLUTIONAL


class ConfigVideoPinballA3c(ConfigBase, ConfigVideoPinball, ConfigA3c):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigVideoPinball.__init__(self)
        ConfigA3c.__init__(self)

        self.MAX_TRAINING_STEPS = 1_000_000
        self.MODEL_TYPE = ModelType.MEDIUM_2D_CONVOLUTIONAL


class ConfigVideoPinballPpo(ConfigBase, ConfigVideoPinball, ConfigPpo):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigVideoPinball.__init__(self)
        ConfigPpo.__init__(self)

        self.MAX_TRAINING_STEPS = 1_000_000
        self.MODEL_TYPE = ModelType.MEDIUM_2D_CONVOLUTIONAL


class ConfigVideoPinballPpoTrajectory(ConfigBase, ConfigVideoPinball, ConfigPpoTrajectory):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigVideoPinball.__init__(self)
        ConfigPpoTrajectory.__init__(self)

        self.MAX_TRAINING_STEPS = 1_000_000
        self.MODEL_TYPE = ModelType.MEDIUM_2D_CONVOLUTIONAL
