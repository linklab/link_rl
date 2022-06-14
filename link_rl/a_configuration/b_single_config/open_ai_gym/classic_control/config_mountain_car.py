from link_rl.a_configuration.a_base_config.b_agents.config_agents_off_policy import ConfigDqn, ConfigDoubleDqn, ConfigDuelingDqn, \
    ConfigDoubleDuelingDqn
from link_rl.a_configuration.a_base_config.b_agents.config_agents_on_policy import ConfigA2c, ConfigReinforce, ConfigPpo, \
    ConfigPpoTrajectory, ConfigA3c
from link_rl.a_configuration.a_base_config.config_single_base import ConfigBase
from link_rl.a_configuration.a_base_config.a_environments.open_ai_gym.config_gym_classic_control import ConfigMountainCar
from link_rl.g_utils.types import ModelType


class ConfigMountainCarDqn(ConfigBase, ConfigMountainCar, ConfigDqn):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigMountainCar.__init__(self)
        ConfigDqn.__init__(self)

        self.LEARNING_RATE = 0.001
        self.MAX_TRAINING_STEPS = 100_000
        self.BUFFER_CAPACITY = 50_000
        self.MODEL_TYPE = ModelType.SMALL_LINEAR


class ConfigMountainCarDoubleDqn(ConfigBase, ConfigMountainCar, ConfigDoubleDqn):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigMountainCar.__init__(self)
        ConfigDoubleDqn.__init__(self)

        self.LEARNING_RATE = 0.001
        self.MAX_TRAINING_STEPS = 100_000
        self.BUFFER_CAPACITY = 50_000
        self.MODEL_TYPE = ModelType.SMALL_LINEAR


class ConfigMountainCarDuelingDqn(ConfigBase, ConfigMountainCar, ConfigDuelingDqn):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigMountainCar.__init__(self)
        ConfigDuelingDqn.__init__(self)

        self.LEARNING_RATE = 0.001
        self.MAX_TRAINING_STEPS = 100_000
        self.BUFFER_CAPACITY = 50_000
        self.MODEL_TYPE = ModelType.SMALL_LINEAR


class ConfigMountainCarDoubleDuelingDqn(ConfigBase, ConfigMountainCar, ConfigDoubleDuelingDqn):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigMountainCar.__init__(self)
        ConfigDoubleDuelingDqn.__init__(self)

        self.LEARNING_RATE = 0.001
        self.MAX_TRAINING_STEPS = 100_000
        self.BUFFER_CAPACITY = 50_000
        self.MODEL_TYPE = ModelType.SMALL_LINEAR


# OnPolicy

class ConfigMountainCarReinforce(ConfigBase, ConfigMountainCar, ConfigReinforce):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigMountainCar.__init__(self)
        ConfigReinforce.__init__(self)

        self.MAX_TRAINING_STEPS = 100_000
        self.MODEL_TYPE = ModelType.SMALL_LINEAR


class ConfigMountainCarA2c(ConfigBase, ConfigMountainCar, ConfigA2c):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigMountainCar.__init__(self)
        ConfigA2c.__init__(self)

        self.MAX_TRAINING_STEPS = 100_000
        self.MODEL_TYPE = ModelType.SMALL_LINEAR


class ConfigMountainCarA3c(ConfigBase, ConfigMountainCar, ConfigA3c):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigMountainCar.__init__(self)
        ConfigA3c.__init__(self)

        self.MAX_TRAINING_STEPS = 100_000
        self.MODEL_TYPE = ModelType.SMALL_LINEAR


class ConfigMountainCarPpo(ConfigBase, ConfigMountainCar, ConfigPpo):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigMountainCar.__init__(self)
        ConfigPpo.__init__(self)

        self.MAX_TRAINING_STEPS = 100_000
        self.MODEL_TYPE = ModelType.SMALL_LINEAR


class ConfigMountainCarPpoTrajectory(ConfigBase, ConfigMountainCar, ConfigPpoTrajectory):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigMountainCar.__init__(self)
        ConfigPpoTrajectory.__init__(self)

        self.MAX_TRAINING_STEPS = 100_000
        self.MODEL_TYPE = ModelType.SMALL_LINEAR
