from a_configuration.a_base_config.a_environments.open_ai_gym.config_gym_box2d import ConfigBipedalWalker
from a_configuration.a_base_config.b_agents.config_agents_off_policy import ConfigDdpg, ConfigSac, ConfigTd3
from a_configuration.a_base_config.b_agents.config_agents_on_policy import ConfigA2c, ConfigPpo, ConfigPpoTrajectory, ConfigA3c
from a_configuration.a_base_config.config_single_base import ConfigBase
from g_utils.types import ModelType, LossFunctionType


class ConfigBipedalWalkerA2c(ConfigBase, ConfigBipedalWalker, ConfigA2c):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigBipedalWalker.__init__(self)
        ConfigA2c.__init__(self)

        self.MAX_TRAINING_STEPS = 200_000
        self.MODEL_TYPE = ModelType.SMALL_LINEAR


class ConfigBipedalWalkerA3c(ConfigBase, ConfigBipedalWalker, ConfigA3c):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigBipedalWalker.__init__(self)
        ConfigA3c.__init__(self)

        self.MAX_TRAINING_STEPS = 200_000
        self.MODEL_TYPE = ModelType.SMALL_LINEAR


class ConfigBipedalWalkerPpo(ConfigBase, ConfigBipedalWalker, ConfigPpo):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigBipedalWalker.__init__(self)
        ConfigPpo.__init__(self)

        self.MAX_TRAINING_STEPS = 200_000
        self.MODEL_TYPE = ModelType.SMALL_LINEAR

        self.USE_GAE = True


class ConfigBipedalWalkerPpoTrajectory(ConfigBase, ConfigBipedalWalker, ConfigPpoTrajectory):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigBipedalWalker.__init__(self)
        ConfigPpoTrajectory.__init__(self)

        self.MAX_TRAINING_STEPS = 500_000
        self.MODEL_TYPE = ModelType.SMALL_LINEAR


class ConfigBipedalWalkerDdpg(ConfigBase, ConfigBipedalWalker, ConfigDdpg):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigBipedalWalker.__init__(self)
        ConfigDdpg.__init__(self)

        self.MAX_TRAINING_STEPS = 300_000
        self.BUFFER_CAPACITY = 200_000
        self.MODEL_TYPE = ModelType.SMALL_LINEAR


class ConfigBipedalWalkerTd3(ConfigBase, ConfigBipedalWalker, ConfigTd3):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigBipedalWalker.__init__(self)
        ConfigTd3.__init__(self)

        self.MAX_TRAINING_STEPS = 300_000
        self.BUFFER_CAPACITY = 200_000
        self.MODEL_TYPE = ModelType.SMALL_LINEAR


class ConfigBipedalWalkerSac(ConfigBase, ConfigBipedalWalker, ConfigSac):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigBipedalWalker.__init__(self)
        ConfigSac.__init__(self)

        self.MAX_TRAINING_STEPS = 300_000
        self.BUFFER_CAPACITY = 200_000
        self.MODEL_TYPE = ModelType.SMALL_LINEAR
