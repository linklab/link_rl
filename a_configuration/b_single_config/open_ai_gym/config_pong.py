from a_configuration.a_base_config.b_agents.agents_off_policy import ConfigDqn, ConfigDoubleDqn, ConfigDuelingDqn, \
    ConfigDoubleDuelingDqn
from a_configuration.a_base_config.a_environments.open_ai_gym.gym_atari import ConfigPong
from a_configuration.a_base_config.b_agents.agents_on_policy import ConfigA2c
from a_configuration.a_base_config.c_models.convolutional_models import ConfigConvolutionalModel
from a_configuration.a_base_config.config_single_base import ConfigBase
from g_utils.types import ModelType


class ConfigPongDqn(
    ConfigBase, ConfigPong, ConfigDqn
):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigPong.__init__(self)
        ConfigDqn.__init__(self)

        self.N_VECTORIZED_ENVS = 1
        self.N_ACTORS = 1
        self.MAX_TRAINING_STEPS = 1_000_000
        self.BUFFER_CAPACITY = 500_000
        self.MODEL_TYPE = ModelType.MEDIUM_CONVOLUTIONAL


class ConfigPongDoubleDqn(
    ConfigBase, ConfigPong, ConfigDoubleDqn
):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigPong.__init__(self)
        ConfigDoubleDqn.__init__(self)

        self.N_VECTORIZED_ENVS = 1
        self.N_ACTORS = 1
        self.MAX_TRAINING_STEPS = 1_000_000
        self.BUFFER_CAPACITY = 500_000
        self.MODEL_TYPE = ModelType.MEDIUM_CONVOLUTIONAL


class ConfigPongDuelingDqn(
    ConfigBase, ConfigPong, ConfigDuelingDqn
):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigPong.__init__(self)
        ConfigDuelingDqn.__init__(self)

        self.N_VECTORIZED_ENVS = 1
        self.N_ACTORS = 1
        self.MAX_TRAINING_STEPS = 1_000_000
        self.BUFFER_CAPACITY = 500_000
        self.MODEL_TYPE = ModelType.MEDIUM_CONVOLUTIONAL


class ConfigPongDoubleDuelingDqn(
    ConfigBase, ConfigPong, ConfigDoubleDuelingDqn
):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigPong.__init__(self)
        ConfigDoubleDuelingDqn.__init__(self)

        self.N_VECTORIZED_ENVS = 1
        self.N_ACTORS = 1
        self.MAX_TRAINING_STEPS = 1_000_000
        self.BUFFER_CAPACITY = 500_000
        self.MODEL_TYPE = ModelType.MEDIUM_CONVOLUTIONAL


class ConfigPongA2c(
    ConfigBase, ConfigPong, ConfigA2c
):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigPong.__init__(self)
        ConfigA2c.__init__(self)

        self.BATCH_SIZE = 64
        self.N_STEP = 1
        self.N_VECTORIZED_ENVS = 2
        self.N_ACTORS = 1
        self.MAX_TRAINING_STEPS = 1_000_000
        self.ENTROPY_BETA = 0.0002
        self.BUFFER_CAPACITY = self.BATCH_SIZE
        self.MODEL_TYPE = ModelType.MEDIUM_CONVOLUTIONAL
