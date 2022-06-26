from link_rl.a_configuration.a_base_config.a_environments.ai_birds.config_ai_birds import ConfigAiBirds
from link_rl.a_configuration.a_base_config.b_agents.config_agents_off_policy import ConfigDqn, ConfigDoubleDqn, \
    ConfigDuelingDqn, ConfigDoubleDuelingDqn
from link_rl.a_configuration.a_base_config.b_agents.config_agents_on_policy import ConfigA3c, ConfigAsynchronousPpo
from link_rl.a_configuration.a_base_config.config_single_base import ConfigBase
from link_rl.g_utils.types import ModelType


class ConfigAiBirdsDqn(ConfigBase, ConfigAiBirds, ConfigDqn):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigAiBirds.__init__(self)
        ConfigDqn.__init__(self)

        self.LEARNING_RATE = 0.001
        self.MAX_TRAINING_STEPS = 1_000_000
        self.BUFFER_CAPACITY = 500_000
        self.MODEL_TYPE = ModelType.MEDIUM_2D_CONVOLUTIONAL

        self.MIN_BUFFER_SIZE_FOR_TRAIN = self.BATCH_SIZE
        self.CONSOLE_LOG_INTERVAL_TRAINING_STEPS = 1
        self.TEST_INTERVAL_TRAINING_STEPS = 10


class ConfigAiBirdsDoubleDqn(ConfigBase, ConfigAiBirds, ConfigDoubleDqn):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigAiBirds.__init__(self)
        ConfigDoubleDqn.__init__(self)

        self.LEARNING_RATE = 0.001
        self.MAX_TRAINING_STEPS = 1_000_000
        self.BUFFER_CAPACITY = 500_000
        self.MODEL_TYPE = ModelType.MEDIUM_2D_CONVOLUTIONAL

        self.MIN_BUFFER_SIZE_FOR_TRAIN = self.BATCH_SIZE
        self.CONSOLE_LOG_INTERVAL_TRAINING_STEPS = 1
        self.TEST_INTERVAL_TRAINING_STEPS = 10


class ConfigAiBirdsDuelingDqn(ConfigBase, ConfigAiBirds, ConfigDuelingDqn):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigAiBirds.__init__(self)
        ConfigDuelingDqn.__init__(self)

        self.LEARNING_RATE = 0.001
        self.MAX_TRAINING_STEPS = 1_000_000
        self.BUFFER_CAPACITY = 500_000
        self.MODEL_TYPE = ModelType.MEDIUM_2D_CONVOLUTIONAL

        self.MIN_BUFFER_SIZE_FOR_TRAIN = self.BATCH_SIZE
        self.CONSOLE_LOG_INTERVAL_TRAINING_STEPS = 1
        self.TEST_INTERVAL_TRAINING_STEPS = 10


class ConfigAiBirdsDoubleDuelingDqn(ConfigBase, ConfigAiBirds, ConfigDoubleDuelingDqn):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigAiBirds.__init__(self)
        ConfigDoubleDuelingDqn.__init__(self)

        self.LEARNING_RATE = 0.001
        self.MAX_TRAINING_STEPS = 1_000_000
        self.BUFFER_CAPACITY = 500_000
        self.MODEL_TYPE = ModelType.MEDIUM_2D_CONVOLUTIONAL

        self.MIN_BUFFER_SIZE_FOR_TRAIN = self.BATCH_SIZE
        self.CONSOLE_LOG_INTERVAL_TRAINING_STEPS = 1
        self.TEST_INTERVAL_TRAINING_STEPS = 10


# OnPolicy
class ConfigAiBirdsA3c(ConfigBase, ConfigAiBirds, ConfigA3c):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigAiBirds.__init__(self)
        ConfigA3c.__init__(self)

        self.MAX_TRAINING_STEPS = 1_000_000
        self.MODEL_TYPE = ModelType.MEDIUM_2D_CONVOLUTIONAL

        self.CONSOLE_LOG_INTERVAL_TRAINING_STEPS = 1
        self.TEST_INTERVAL_TRAINING_STEPS = 10


class ConfigAiBirdsAsynchronousPpo(ConfigBase, ConfigAiBirds, ConfigAsynchronousPpo):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigAiBirds.__init__(self)
        ConfigAsynchronousPpo.__init__(self)

        self.MAX_TRAINING_STEPS = 1_000_000
        self.MODEL_TYPE = ModelType.MEDIUM_2D_CONVOLUTIONAL

        self.CONSOLE_LOG_INTERVAL_TRAINING_STEPS = 1
        self.TEST_INTERVAL_TRAINING_STEPS = 10
