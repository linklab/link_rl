from a_configuration.a_base_config.a_environments.open_ai_gym.gym_box2d import ConfigLunarLanderContinuous
from a_configuration.a_base_config.b_agents.agents_off_policy import ConfigDdpg, ConfigSac, ConfigTd3
from a_configuration.a_base_config.b_agents.agents_on_policy import ConfigA2c, ConfigPpo, ConfigPpoTrajectory
from a_configuration.a_base_config.config_single_base import ConfigBase
from g_utils.types import ModelType


class ConfigLunarLanderContinuousA2c(ConfigBase, ConfigLunarLanderContinuous, ConfigA2c):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigLunarLanderContinuous.__init__(self)
        ConfigA2c.__init__(self)

        self.MAX_TRAINING_STEPS = 200_000
        self.MODEL_TYPE = ModelType.SMALL_LINEAR


class ConfigLunarLanderContinuousPpo(ConfigBase, ConfigLunarLanderContinuous, ConfigPpo):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigLunarLanderContinuous.__init__(self)
        ConfigPpo.__init__(self)

        self.MAX_TRAINING_STEPS = 200_000
        self.MODEL_TYPE = ModelType.SMALL_LINEAR


class ConfigLunarLanderContinuousPpoTrajectory(ConfigBase, ConfigLunarLanderContinuous, ConfigPpoTrajectory):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigLunarLanderContinuous.__init__(self)
        ConfigPpoTrajectory.__init__(self)

        self.MAX_TRAINING_STEPS = 500_000
        self.MODEL_TYPE = ModelType.SMALL_LINEAR


class ConfigLunarLanderContinuousDdpg(ConfigBase, ConfigLunarLanderContinuous, ConfigDdpg):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigLunarLanderContinuous.__init__(self)
        ConfigDdpg.__init__(self)

        self.MAX_TRAINING_STEPS = 300_000
        self.BUFFER_CAPACITY = 200_000
        self.MODEL_TYPE = ModelType.SMALL_LINEAR


class ConfigLunarLanderContinuousTd3(ConfigBase, ConfigLunarLanderContinuous, ConfigTd3):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigLunarLanderContinuous.__init__(self)
        ConfigTd3.__init__(self)

        self.MAX_TRAINING_STEPS = 300_000
        self.BUFFER_CAPACITY = 200_000
        self.MODEL_TYPE = ModelType.SMALL_LINEAR


class ConfigLunarLanderContinuousSac(ConfigBase, ConfigLunarLanderContinuous, ConfigSac):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigLunarLanderContinuous.__init__(self)
        ConfigSac.__init__(self)

        self.MAX_TRAINING_STEPS = 300_000
        self.BUFFER_CAPACITY = 200_000
        self.MODEL_TYPE = ModelType.SMALL_LINEAR
