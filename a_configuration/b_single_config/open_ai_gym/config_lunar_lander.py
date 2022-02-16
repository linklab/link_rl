from a_configuration.a_base_config.a_environments.open_ai_gym.config_gym_box2d import ConfigLunarLander
from a_configuration.a_base_config.b_agents.config_agents_off_policy import ConfigDqn, ConfigMuzero
from a_configuration.a_base_config.b_agents.config_agents_on_policy import ConfigA2c, ConfigPpo, ConfigPpoTrajectory, ConfigA3c
from a_configuration.a_base_config.config_single_base import ConfigBase
from g_utils.types import ModelType


class ConfigLunarLanderDqn(ConfigBase, ConfigLunarLander, ConfigDqn):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigLunarLander.__init__(self)
        ConfigDqn.__init__(self)

        self.MAX_TRAINING_STEPS = 100_000
        self.BUFFER_CAPACITY = 100_000
        self.LEARNING_RATE = 0.001
        self.MODEL_TYPE = ModelType.SMALL_LINEAR


class ConfigLunarLanderMuzero(ConfigBase, ConfigLunarLander, ConfigMuzero):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigLunarLander.__init__(self)
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


class ConfigLunarLanderA2c(ConfigBase, ConfigLunarLander, ConfigA2c):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigLunarLander.__init__(self)
        ConfigA2c.__init__(self)

        self.MAX_TRAINING_STEPS = 100_000
        self.MODEL_TYPE = ModelType.SMALL_LINEAR


class ConfigLunarLanderA3c(ConfigBase, ConfigLunarLander, ConfigA3c):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigLunarLander.__init__(self)
        ConfigA3c.__init__(self)

        self.MAX_TRAINING_STEPS = 100_000
        self.MODEL_TYPE = ModelType.SMALL_LINEAR


class ConfigLunarLanderPpo(ConfigBase, ConfigLunarLander, ConfigPpo):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigLunarLander.__init__(self)
        ConfigPpo.__init__(self)

        self.MAX_TRAINING_STEPS = 100_000
        self.MODEL_TYPE = ModelType.SMALL_LINEAR


class ConfigLunarLanderPpoTrajectory(ConfigBase, ConfigLunarLander, ConfigPpoTrajectory):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigLunarLander.__init__(self)
        ConfigPpoTrajectory.__init__(self)

        self.MAX_TRAINING_STEPS = 100_000
        self.MODEL_TYPE = ModelType.SMALL_LINEAR
