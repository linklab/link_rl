from link_rl.a_configuration.a_base_config.b_agents.config_agents_off_policy import ConfigDqn, ConfigDoubleDqn, \
    ConfigDuelingDqn, ConfigDoubleDuelingDqn
from link_rl.a_configuration.a_base_config.config_single_base import ConfigBase
from link_rl.a_configuration.a_base_config.a_environments.task_allocation.config_basic_task_allocation import ConfigBasicTaskAllocation0


class ConfigBasicTaskAllocation0Dqn(ConfigBase, ConfigBasicTaskAllocation0, ConfigDqn):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigBasicTaskAllocation0.__init__(self)
        ConfigDqn.__init__(self)

        self.NUM_TASK = 20
        self.NUM_RESOURCES = 1
        self.INITIAL_RESOURCES_CAPACITY = [200]
        self.LOW_DEMAND_RESOURCE_AT_TASK = [10]
        self.HIGH_DEMAND_RESOURCE_AT_TASK = [15]
        self.INITIAL_TASK_DISTRIBUTION_FIXED = True
        self.MAX_TRAINING_STEPS = self.NUM_TASK * 2_000
        self.BUFFER_CAPACITY = self.NUM_TASK * 1_000


class ConfigBasicTaskAllocation1Dqn(ConfigBase, ConfigBasicTaskAllocation0, ConfigDqn):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigBasicTaskAllocation0.__init__(self)
        ConfigDqn.__init__(self)

        self.NUM_TASK = 20
        self.NUM_RESOURCES = 2
        self.INITIAL_RESOURCES_CAPACITY = [100, 100]
        self.LOW_DEMAND_RESOURCE_AT_TASK = [10]
        self.HIGH_DEMAND_RESOURCE_AT_TASK = [15]
        self.INITIAL_TASK_DISTRIBUTION_FIXED = True
        self.MAX_TRAINING_STEPS = self.NUM_TASK * 2_000
        self.BUFFER_CAPACITY = self.NUM_TASK * 1_000


class ConfigBasicTaskAllocation0DoubleDqn(ConfigBase, ConfigBasicTaskAllocation0, ConfigDoubleDqn):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigBasicTaskAllocation0.__init__(self)
        ConfigDoubleDqn.__init__(self)

        self.NUM_TASK = 20
        self.NUM_RESOURCES = 1
        self.INITIAL_RESOURCES_CAPACITY = [200]
        self.LOW_DEMAND_RESOURCE_AT_TASK = [10]
        self.HIGH_DEMAND_RESOURCE_AT_TASK = [15]
        self.INITIAL_TASK_DISTRIBUTION_FIXED = True
        self.MAX_TRAINING_STEPS = self.NUM_TASK * 2_000
        self.BUFFER_CAPACITY = self.NUM_TASK * 1_000


class ConfigBasicTaskAllocation0DuelingDqn(ConfigBase, ConfigBasicTaskAllocation0, ConfigDuelingDqn):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigBasicTaskAllocation0.__init__(self)
        ConfigDuelingDqn.__init__(self)

        self.NUM_TASK = 20
        self.NUM_RESOURCES = 1
        self.INITIAL_RESOURCES_CAPACITY = [200]
        self.LOW_DEMAND_RESOURCE_AT_TASK = [10]
        self.HIGH_DEMAND_RESOURCE_AT_TASK = [15]
        self.INITIAL_TASK_DISTRIBUTION_FIXED = True
        self.MAX_TRAINING_STEPS = self.NUM_TASK * 2_000
        self.BUFFER_CAPACITY = self.NUM_TASK * 1_000


class ConfigBasicTaskAllocation0DoubleDuelingDqn(ConfigBase, ConfigBasicTaskAllocation0, ConfigDoubleDuelingDqn):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigBasicTaskAllocation0.__init__(self)
        ConfigDoubleDuelingDqn.__init__(self)

        self.NUM_TASK = 20
        self.NUM_RESOURCES = 1
        self.INITIAL_RESOURCES_CAPACITY = [200]
        self.LOW_DEMAND_RESOURCE_AT_TASK = [10]
        self.HIGH_DEMAND_RESOURCE_AT_TASK = [15]
        self.INITIAL_TASK_DISTRIBUTION_FIXED = True
        self.MAX_TRAINING_STEPS = self.NUM_TASK * 2_000
        self.BUFFER_CAPACITY = self.NUM_TASK * 1_000
