from link_rl.a_configuration.a_base_config.b_agents.config_agents_off_policy import ConfigDqn, ConfigDoubleDqn, ConfigDuelingDqn, \
    ConfigDoubleDuelingDqn
from link_rl.a_configuration.a_base_config.b_agents.config_agents_on_policy import ConfigA2c, ConfigReinforce, ConfigPpo, \
    ConfigPpoTrajectory, ConfigA3c
from link_rl.a_configuration.a_base_config.config_single_base import ConfigBase
from link_rl.a_configuration.a_base_config.a_environments.open_ai_gym.config_gym_classic_control import ConfigAcrobot
from link_rl.g_utils.types import ModelType


class ConfigAcrobotDqn(ConfigBase, ConfigAcrobot, ConfigDqn):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigAcrobot.__init__(self)
        ConfigDqn.__init__(self)

        self.LEARNING_RATE = 0.001
        self.MAX_TRAINING_STEPS = 100_000
        self.BUFFER_CAPACITY = 50_000
        self.MODEL_TYPE = ModelType.SMALL_LINEAR


class ConfigAcrobotDoubleDqn(ConfigBase, ConfigAcrobot, ConfigDoubleDqn):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigAcrobot.__init__(self)
        ConfigDoubleDqn.__init__(self)

        self.LEARNING_RATE = 0.001
        self.MAX_TRAINING_STEPS = 100_000
        self.BUFFER_CAPACITY = 50_000
        self.MODEL_TYPE = ModelType.SMALL_LINEAR


class ConfigAcrobotDuelingDqn(ConfigBase, ConfigAcrobot, ConfigDuelingDqn):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigAcrobot.__init__(self)
        ConfigDuelingDqn.__init__(self)

        self.LEARNING_RATE = 0.001
        self.MAX_TRAINING_STEPS = 100_000
        self.BUFFER_CAPACITY = 50_000
        self.MODEL_TYPE = ModelType.SMALL_LINEAR


class ConfigAcrobotDoubleDuelingDqn(ConfigBase, ConfigAcrobot, ConfigDoubleDuelingDqn):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigAcrobot.__init__(self)
        ConfigDoubleDuelingDqn.__init__(self)

        self.LEARNING_RATE = 0.001
        self.MAX_TRAINING_STEPS = 100_000
        self.BUFFER_CAPACITY = 50_000
        self.MODEL_TYPE = ModelType.SMALL_LINEAR


# OnPolicy

class ConfigAcrobotReinforce(ConfigBase, ConfigAcrobot, ConfigReinforce):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigAcrobot.__init__(self)
        ConfigReinforce.__init__(self)

        self.MAX_TRAINING_STEPS = 100_000
        self.MODEL_TYPE = ModelType.SMALL_LINEAR


class ConfigAcrobotA2c(ConfigBase, ConfigAcrobot, ConfigA2c):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigAcrobot.__init__(self)
        ConfigA2c.__init__(self)

        self.MAX_TRAINING_STEPS = 100_000
        self.MODEL_TYPE = ModelType.SMALL_LINEAR


class ConfigAcrobotA3c(ConfigBase, ConfigAcrobot, ConfigA3c):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigAcrobot.__init__(self)
        ConfigA3c.__init__(self)

        self.MAX_TRAINING_STEPS = 100_000
        self.MODEL_TYPE = ModelType.SMALL_LINEAR


class ConfigAcrobotPpo(ConfigBase, ConfigAcrobot, ConfigPpo):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigAcrobot.__init__(self)
        ConfigPpo.__init__(self)

        self.MAX_TRAINING_STEPS = 100_000
        self.MODEL_TYPE = ModelType.SMALL_LINEAR


class ConfigAcrobotPpoTrajectory(ConfigBase, ConfigAcrobot, ConfigPpoTrajectory):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigAcrobot.__init__(self)
        ConfigPpoTrajectory.__init__(self)

        self.MAX_TRAINING_STEPS = 100_000
        self.MODEL_TYPE = ModelType.SMALL_LINEAR
