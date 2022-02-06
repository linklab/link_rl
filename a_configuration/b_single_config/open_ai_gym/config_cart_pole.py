from a_configuration.a_base_config.b_agents.agents_off_policy import ConfigDqn, ConfigDoubleDqn, ConfigDuelingDqn, \
    ConfigDoubleDuelingDqn, ConfigSac, ConfigMuzero
from a_configuration.a_base_config.b_agents.agents_on_policy import ConfigA2c, ConfigReinforce, ConfigPpo, \
    ConfigPpoTrajectory
from a_configuration.a_base_config.config_single_base import ConfigBase
from a_configuration.a_base_config.a_environments.open_ai_gym.gym_classic_control import ConfigCartPole
from g_utils.types import ModelType


class ConfigCartPoleDqn(ConfigBase, ConfigCartPole, ConfigDqn):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigCartPole.__init__(self)
        ConfigDqn.__init__(self)

        self.MAX_TRAINING_STEPS = 100_000
        self.BUFFER_CAPACITY = 50_000
        self.MODEL_TYPE = ModelType.SMALL_LINEAR


class ConfigCartPoleDoubleDqn(ConfigBase, ConfigCartPole, ConfigDoubleDqn):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigCartPole.__init__(self)
        ConfigDoubleDqn.__init__(self)

        self.N_VECTORIZED_ENVS = 1
        self.N_ACTORS = 1
        self.MAX_TRAINING_STEPS = 100_000
        self.BUFFER_CAPACITY = 50_000
        self.MODEL_TYPE = ModelType.SMALL_LINEAR


class ConfigCartPoleDuelingDqn(ConfigBase, ConfigCartPole, ConfigDuelingDqn):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigCartPole.__init__(self)
        ConfigDuelingDqn.__init__(self)

        self.N_VECTORIZED_ENVS = 1
        self.N_ACTORS = 1
        self.MAX_TRAINING_STEPS = 100_000
        self.BUFFER_CAPACITY = 50_000
        self.MODEL_TYPE = ModelType.SMALL_LINEAR


class ConfigCartPoleDoubleDuelingDqn(ConfigBase, ConfigCartPole, ConfigDoubleDuelingDqn):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigCartPole.__init__(self)
        ConfigDoubleDuelingDqn.__init__(self)

        self.N_VECTORIZED_ENVS = 1
        self.N_ACTORS = 1
        self.MAX_TRAINING_STEPS = 100_000
        self.BUFFER_CAPACITY = 50_000
        self.MODEL_TYPE = ModelType.SMALL_LINEAR

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
        self.BATCH_SIZE = 256
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


class ConfigCartPoleMuzero(ConfigBase, ConfigCartPole, ConfigMuzero):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigCartPole.__init__(self)
        ConfigMuzero.__init__(self)

        self.MAX_TRAINING_STEPS = 100_000
        self.MODEL_TYPE = ModelType.SMALL_LINEAR
        self.BUFFER_CAPACITY = 10_000