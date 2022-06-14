from link_rl.a_configuration.a_base_config.b_agents.config_agents_off_policy import ConfigDqn, ConfigDoubleDqn, ConfigDuelingDqn, \
    ConfigDoubleDuelingDqn, ConfigMuzero
from link_rl.a_configuration.a_base_config.b_agents.config_agents_on_policy import ConfigA2c, ConfigReinforce, ConfigPpo, \
    ConfigPpoTrajectory, ConfigA3c
from link_rl.a_configuration.a_base_config.config_single_base import ConfigBase
from link_rl.a_configuration.a_base_config.a_environments.open_ai_gym.config_gym_classic_control import ConfigCartPole
from link_rl.g_utils.types import ModelType


class ConfigCartPoleDqn(ConfigBase, ConfigCartPole, ConfigDqn):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigCartPole.__init__(self)
        ConfigDqn.__init__(self)

        self.LEARNING_RATE = 0.001
        self.MAX_TRAINING_STEPS = 100_000
        self.BUFFER_CAPACITY = 50_000
        self.MODEL_TYPE = ModelType.SMALL_LINEAR


class ConfigCartPoleDoubleDqn(ConfigBase, ConfigCartPole, ConfigDoubleDqn):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigCartPole.__init__(self)
        ConfigDoubleDqn.__init__(self)

        self.LEARNING_RATE = 0.001
        self.MAX_TRAINING_STEPS = 100_000
        self.BUFFER_CAPACITY = 50_000
        self.MODEL_TYPE = ModelType.SMALL_LINEAR


class ConfigCartPoleDuelingDqn(ConfigBase, ConfigCartPole, ConfigDuelingDqn):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigCartPole.__init__(self)
        ConfigDuelingDqn.__init__(self)

        self.LEARNING_RATE = 0.001
        self.MAX_TRAINING_STEPS = 100_000
        self.BUFFER_CAPACITY = 50_000
        self.MODEL_TYPE = ModelType.SMALL_LINEAR


class ConfigCartPoleDoubleDuelingDqn(ConfigBase, ConfigCartPole, ConfigDoubleDuelingDqn):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigCartPole.__init__(self)
        ConfigDoubleDuelingDqn.__init__(self)

        self.LEARNING_RATE = 0.001
        self.MAX_TRAINING_STEPS = 100_000
        self.BUFFER_CAPACITY = 50_000
        self.MODEL_TYPE = ModelType.SMALL_LINEAR


class ConfigCartPoleMuzero(ConfigBase, ConfigCartPole, ConfigMuzero):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigCartPole.__init__(self)
        ConfigMuzero.__init__(self)

        self.MAX_TRAINING_STEPS = 10_000
        self.MODEL_TYPE = ModelType.SMALL_LINEAR
        self.BUFFER_CAPACITY = 500
        self.SUPPORT_SIZE = 10
        self.CONSOLE_LOG_INTERVAL_TRAINING_STEPS = 30
        self.VALUE_LOSS_WEIGHT = 1
        self.TRAIN_INTERVAL_GLOBAL_TIME_STEPS = 10
        self.TEST_INTERVAL_TRAINING_STEPS = 100
        self.NUM_UNROLL_STEPS = 10
        self.N_STEP = 50

# OnPolicy


class ConfigCartPoleReinforce(ConfigBase, ConfigCartPole, ConfigReinforce):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigCartPole.__init__(self)
        ConfigReinforce.__init__(self)

        self.MAX_TRAINING_STEPS = 100_000
        self.MODEL_TYPE = ModelType.SMALL_LINEAR


class ConfigCartPoleA2c(ConfigBase, ConfigCartPole, ConfigA2c):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigCartPole.__init__(self)
        ConfigA2c.__init__(self)

        self.MAX_TRAINING_STEPS = 100_000
        self.MODEL_TYPE = ModelType.SMALL_LINEAR


class ConfigCartPoleA3c(ConfigBase, ConfigCartPole, ConfigA3c):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigCartPole.__init__(self)
        ConfigA3c.__init__(self)

        self.MAX_TRAINING_STEPS = 100_000
        self.MODEL_TYPE = ModelType.SMALL_LINEAR


class ConfigCartPolePpo(ConfigBase, ConfigCartPole, ConfigPpo):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigCartPole.__init__(self)
        ConfigPpo.__init__(self)

        self.MAX_TRAINING_STEPS = 100_000
        self.MODEL_TYPE = ModelType.SMALL_LINEAR


class ConfigCartPolePpoTrajectory(ConfigBase, ConfigCartPole, ConfigPpoTrajectory):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigCartPole.__init__(self)
        ConfigPpoTrajectory.__init__(self)

        self.MAX_TRAINING_STEPS = 100_000
        self.MODEL_TYPE = ModelType.SMALL_LINEAR



