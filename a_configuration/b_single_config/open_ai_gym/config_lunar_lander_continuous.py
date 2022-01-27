from a_configuration.a_base_config.a_environments.open_ai_gym.gym_box2d import ConfigLunarLanderContinuous
from a_configuration.a_base_config.b_agents.agents_off_policy import ConfigDdpg, ConfigSac
from a_configuration.a_base_config.b_agents.agents_on_policy import ConfigA2c, ConfigPpo
from a_configuration.a_base_config.config_single_base import ConfigBase
from g_utils.types import ModelType


class ConfigLunarLanderContinuousA2c(ConfigBase, ConfigLunarLanderContinuous, ConfigA2c):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigLunarLanderContinuous.__init__(self)
        ConfigA2c.__init__(self)

        self.N_VECTORIZED_ENVS = 1
        self.N_ACTORS = 1
        self.MAX_TRAINING_STEPS = 200_000
        self.BUFFER_CAPACITY = 200_000
        self.BATCH_SIZE = 64
        self.MODEL_TYPE = ModelType.SMALL_LINEAR


class ConfigLunarLanderContinuousPpo(ConfigBase, ConfigLunarLanderContinuous, ConfigPpo):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigLunarLanderContinuous.__init__(self)
        ConfigPpo.__init__(self)

        self.N_VECTORIZED_ENVS = 1
        self.N_ACTORS = 1
        self.MAX_TRAINING_STEPS = 200_000
        self.MODEL_TYPE = ModelType.SMALL_LINEAR

        self.BATCH_SIZE = 256
        self.PPO_TRAJECTORY_SIZE = self.BATCH_SIZE * 10
        self.PPO_K_EPOCH = 3
        self.BUFFER_CAPACITY = self.PPO_TRAJECTORY_SIZE
        self.CONSOLE_LOG_INTERVAL_TRAINING_STEPS = 10 * 3


class ConfigLunarLanderContinuousDdpg(ConfigBase, ConfigLunarLanderContinuous, ConfigDdpg):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigLunarLanderContinuous.__init__(self)
        ConfigDdpg.__init__(self)

        self.N_VECTORIZED_ENVS = 1
        self.N_ACTORS = 1
        self.MAX_TRAINING_STEPS = 200_000
        self.BUFFER_CAPACITY = 200_000
        self.BATCH_SIZE = 64
        self.LEARNING_RATE = 0.001
        self.MODEL_TYPE = ModelType.SMALL_LINEAR


class ConfigLunarLanderContinuousSac(ConfigBase, ConfigLunarLanderContinuous, ConfigSac):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigLunarLanderContinuous.__init__(self)
        ConfigSac.__init__(self)

        self.N_VECTORIZED_ENVS = 1
        self.N_ACTORS = 1
        self.MAX_TRAINING_STEPS = 500_000
        self.BUFFER_CAPACITY = 200_000
        self.BATCH_SIZE = 64
        self.MODEL_TYPE = ModelType.SMALL_LINEAR
