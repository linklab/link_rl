from a_configuration.a_base_config.a_environments.pybullet.config_gym_pybullet import ConfigCartPoleBullet
from a_configuration.a_base_config.b_agents.config_agents_off_policy import ConfigDqn, \
    ConfigDoubleDqn, ConfigDuelingDqn, ConfigDoubleDuelingDqn
from a_configuration.a_base_config.b_agents.config_agents_on_policy import ConfigA2c, ConfigPpo, ConfigPpoTrajectory, \
    ConfigA3c, ConfigReinforce
from a_configuration.a_base_config.config_single_base import ConfigBase
from g_utils.types import ModelType


class ConfigCartPoleBulletDqn(ConfigBase, ConfigCartPoleBullet, ConfigDqn):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigCartPoleBullet.__init__(self)
        ConfigDqn.__init__(self)

        self.MAX_TRAINING_STEPS = 100_000
        self.MODEL_TYPE = ModelType.SMALL_LINEAR


class ConfigCartPoleBulletDoubleDqn(ConfigBase, ConfigCartPoleBullet, ConfigDoubleDqn):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigCartPoleBullet.__init__(self)
        ConfigDoubleDqn.__init__(self)

        self.MAX_TRAINING_STEPS = 100_000
        self.MODEL_TYPE = ModelType.SMALL_LINEAR


class ConfigCartPoleBulletDuelingDqn(ConfigBase, ConfigCartPoleBullet, ConfigDuelingDqn):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigCartPoleBullet.__init__(self)
        ConfigDuelingDqn.__init__(self)

        self.MAX_TRAINING_STEPS = 100_000
        self.MODEL_TYPE = ModelType.SMALL_LINEAR


class ConfigCartPoleBulletDoubleDuelingDqn(ConfigBase, ConfigCartPoleBullet, ConfigDoubleDuelingDqn):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigCartPoleBullet.__init__(self)
        ConfigDoubleDuelingDqn.__init__(self)

        self.MAX_TRAINING_STEPS = 100_000
        self.MODEL_TYPE = ModelType.SMALL_LINEAR


class ConfigCartPoleBulletReinforce(ConfigBase, ConfigCartPoleBullet, ConfigReinforce):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigCartPoleBullet.__init__(self)
        ConfigReinforce.__init__(self)

        self.MAX_TRAINING_STEPS = 100_000
        self.MODEL_TYPE = ModelType.SMALL_LINEAR


class ConfigCartPoleBulletA2c(ConfigBase, ConfigCartPoleBullet, ConfigA2c):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigCartPoleBullet.__init__(self)
        ConfigA2c.__init__(self)

        self.MAX_TRAINING_STEPS = 100_000
        self.MODEL_TYPE = ModelType.SMALL_LINEAR


class ConfigCartPoleBulletA3c(ConfigBase, ConfigCartPoleBullet, ConfigA3c):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigCartPoleBullet.__init__(self)
        ConfigA3c.__init__(self)

        self.MAX_TRAINING_STEPS = 100_000
        self.MODEL_TYPE = ModelType.SMALL_LINEAR


class ConfigCartPoleBulletPpo(ConfigBase, ConfigCartPoleBullet, ConfigPpo):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigCartPoleBullet.__init__(self)
        ConfigPpo.__init__(self)

        self.MAX_TRAINING_STEPS = 100_000
        self.MODEL_TYPE = ModelType.SMALL_LINEAR


class ConfigCartPoleBulletPpoTrajectory(ConfigBase, ConfigCartPoleBullet, ConfigPpoTrajectory):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigCartPoleBullet.__init__(self)
        ConfigPpoTrajectory.__init__(self)

        self.MAX_TRAINING_STEPS = 100_000
        self.MODEL_TYPE = ModelType.SMALL_LINEAR
