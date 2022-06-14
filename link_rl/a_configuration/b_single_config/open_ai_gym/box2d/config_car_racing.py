from link_rl.a_configuration.a_base_config.a_environments.open_ai_gym.config_gym_box2d import ConfigCarRacing
from link_rl.a_configuration.a_base_config.b_agents.config_agents_off_policy import ConfigDdpg, ConfigSac, ConfigTd3
from link_rl.a_configuration.a_base_config.b_agents.config_agents_on_policy import ConfigA2c, ConfigPpo, ConfigPpoTrajectory, ConfigA3c
from link_rl.a_configuration.a_base_config.config_single_base import ConfigBase
from link_rl.g_utils.types import ModelType


class ConfigCarRacingA2c(ConfigBase, ConfigCarRacing, ConfigA2c):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigCarRacing.__init__(self)
        ConfigA2c.__init__(self)

        self.MAX_TRAINING_STEPS = 200_000
        self.MODEL_TYPE = ModelType.SMALL_2D_CONVOLUTIONAL


class ConfigCarRacingA3c(ConfigBase, ConfigCarRacing, ConfigA3c):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigCarRacing.__init__(self)
        ConfigA3c.__init__(self)

        self.MAX_TRAINING_STEPS = 200_000
        self.MODEL_TYPE = ModelType.SMALL_2D_CONVOLUTIONAL


class ConfigCarRacingPpo(ConfigBase, ConfigCarRacing, ConfigPpo):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigCarRacing.__init__(self)
        ConfigPpo.__init__(self)

        self.MAX_TRAINING_STEPS = 200_000
        self.MODEL_TYPE = ModelType.SMALL_2D_CONVOLUTIONAL

        self.USE_GAE = True


class ConfigCarRacingPpoTrajectory(ConfigBase, ConfigCarRacing, ConfigPpoTrajectory):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigCarRacing.__init__(self)
        ConfigPpoTrajectory.__init__(self)

        self.MAX_TRAINING_STEPS = 500_000
        self.MODEL_TYPE = ModelType.SMALL_2D_CONVOLUTIONAL


class ConfigCarRacingDdpg(ConfigBase, ConfigCarRacing, ConfigDdpg):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigCarRacing.__init__(self)
        ConfigDdpg.__init__(self)

        self.MAX_TRAINING_STEPS = 300_000
        self.BUFFER_CAPACITY = 200_000
        self.MODEL_TYPE = ModelType.SMALL_2D_CONVOLUTIONAL


class ConfigCarRacingTd3(ConfigBase, ConfigCarRacing, ConfigTd3):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigCarRacing.__init__(self)
        ConfigTd3.__init__(self)

        self.MAX_TRAINING_STEPS = 300_000
        self.BUFFER_CAPACITY = 200_000
        self.MODEL_TYPE = ModelType.SMALL_2D_CONVOLUTIONAL


class ConfigCarRacingSac(ConfigBase, ConfigCarRacing, ConfigSac):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigCarRacing.__init__(self)
        ConfigSac.__init__(self)

        self.MAX_TRAINING_STEPS = 300_000
        self.BUFFER_CAPACITY = 200_000
        self.MODEL_TYPE = ModelType.SMALL_2D_CONVOLUTIONAL
