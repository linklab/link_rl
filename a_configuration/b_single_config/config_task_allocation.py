from a_configuration.a_base_config.b_agents.config_agents_off_policy import ConfigDqn, ConfigDoubleDqn, \
    ConfigDuelingDqn, ConfigDoubleDuelingDqn
from a_configuration.a_base_config.config_single_base import ConfigBase
from a_configuration.a_base_config.a_environments.config_task_allocation import ConfigTakAllocation0
from g_utils.types import ModelType


class ConfigTaskAllocation0Dqn(ConfigBase, ConfigTakAllocation0, ConfigDqn):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigTakAllocation0.__init__(self)
        ConfigDqn.__init__(self)

        self.LEARNING_RATE = 0.0003
        self.MAX_TRAINING_STEPS = 200_000
        self.BUFFER_CAPACITY = 35_000
        self.CONSOLE_LOG_INTERVAL_TRAINING_STEPS = 20
        self.MODEL_TYPE = ModelType.SMALL_LINEAR
        self.TARGET_SYNC_INTERVAL_TRAINING_STEPS = 100


class ConfigTaskAllocation0DoubleDqn(ConfigBase, ConfigTakAllocation0, ConfigDoubleDqn):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigTakAllocation0.__init__(self)
        ConfigDoubleDqn.__init__(self)

        self.LEARNING_RATE = 0.0003
        self.MAX_TRAINING_STEPS = 200_000
        self.BUFFER_CAPACITY = 35_000
        self.CONSOLE_LOG_INTERVAL_TRAINING_STEPS = 20
        self.MODEL_TYPE = ModelType.SMALL_LINEAR
        self.TARGET_SYNC_INTERVAL_TRAINING_STEPS = 100


class ConfigTaskAllocation0DuelingDqn(ConfigBase, ConfigTakAllocation0, ConfigDuelingDqn):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigTakAllocation0.__init__(self)
        ConfigDuelingDqn.__init__(self)

        self.LEARNING_RATE = 0.0003
        self.MAX_TRAINING_STEPS = 200_000
        self.BUFFER_CAPACITY = 35_000
        self.CONSOLE_LOG_INTERVAL_TRAINING_STEPS = 20
        self.MODEL_TYPE = ModelType.SMALL_LINEAR
        self.TARGET_SYNC_INTERVAL_TRAINING_STEPS = 100


class ConfigTaskAllocation0DoubleDuelingDqn(ConfigBase, ConfigTakAllocation0, ConfigDoubleDuelingDqn):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigTakAllocation0.__init__(self)
        ConfigDoubleDuelingDqn.__init__(self)

        self.LEARNING_RATE = 0.0003
        self.MAX_TRAINING_STEPS = 200_000
        self.BUFFER_CAPACITY = 35_000
        self.CONSOLE_LOG_INTERVAL_TRAINING_STEPS = 20
        self.MODEL_TYPE = ModelType.SMALL_LINEAR
        self.TARGET_SYNC_INTERVAL_TRAINING_STEPS = 100
