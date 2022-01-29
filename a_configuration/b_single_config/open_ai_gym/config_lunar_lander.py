from a_configuration.a_base_config.a_environments.open_ai_gym.gym_box2d import ConfigLunarLander
from a_configuration.a_base_config.b_agents.agents_off_policy import ConfigDqn
from a_configuration.a_base_config.b_agents.agents_on_policy import ConfigA2c, ConfigPpo, ConfigPpoTrajectory
from a_configuration.a_base_config.config_single_base import ConfigBase
from g_utils.types import ModelType


class ConfigLunarLanderDqn(ConfigBase, ConfigLunarLander, ConfigDqn):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigLunarLander.__init__(self)
        ConfigDqn.__init__(self)

        self.N_VECTORIZED_ENVS = 1
        self.N_ACTORS = 1
        self.MAX_TRAINING_STEPS = 100_000
        self.BUFFER_CAPACITY = 100_000
        self.BATCH_SIZE = 64
        self.LEARNING_RATE = 0.001
        self.MODEL_TYPE = ModelType.SMALL_LINEAR


class ConfigLunarLanderA2c(ConfigBase, ConfigLunarLander, ConfigA2c):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigLunarLander.__init__(self)
        ConfigA2c.__init__(self)

        self.MAX_TRAINING_STEPS = 100_000
        self.BUFFER_CAPACITY = 100_000
        self.BATCH_SIZE = 256
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
