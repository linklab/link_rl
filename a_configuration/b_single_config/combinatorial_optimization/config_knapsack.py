import os

from a_configuration.a_base_config.b_agents.config_agents_off_policy import ConfigDqn
from a_configuration.a_base_config.b_agents.config_agents_on_policy import ConfigA2c, ConfigPpo
from a_configuration.a_base_config.config_single_base import ConfigBase
from a_configuration.a_base_config.a_environments.combinatorial_optimization.config_knapsack import ConfigKnapsack0


class ConfigKnapsack0Dqn(ConfigBase, ConfigKnapsack0, ConfigDqn):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigKnapsack0.__init__(self)
        ConfigDqn.__init__(self)

        self.NUM_ITEM = 20
        self.LIMIT_WEIGHT_KNAPSACK = 200
        self.MIN_WEIGHT_ITEM = 10
        self.MAX_WEIGHT_ITEM = 15
        self.MIN_VALUE_ITEM = 10
        self.MAX_VALUE_ITEM = 15
        self.INITIAL_TASK_DISTRIBUTION_FIXED = True
        self.MAX_TRAINING_STEPS = self.NUM_ITEM * 2_000
        self.BUFFER_CAPACITY = self.NUM_ITEM * 1_000


class ConfigKnapsackRandomInsatances0Dqn(ConfigBase, ConfigKnapsack0, ConfigDqn):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigKnapsack0.__init__(self)
        ConfigDqn.__init__(self)

        self.FILE_PATH = os.path.join(self.PROJECT_HOME, "b_environments\\combinatorial_optimization\\random_instances\\n_50_r_100\\instance0.csv")
        self.NUM_ITEM = 20
        self.INITIAL_TASK_DISTRIBUTION_FIXED = True
        self.MAX_TRAINING_STEPS = self.NUM_ITEM * 2_000
        self.BUFFER_CAPACITY = self.NUM_ITEM * 1_000


class ConfigKnapsack0A2c(ConfigBase, ConfigKnapsack0, ConfigA2c):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigKnapsack0.__init__(self)
        ConfigA2c.__init__(self)

        self.NUM_ITEM = 20
        self.LIMIT_WEIGHT_KNAPSACK = 200
        self.MIN_WEIGHT_ITEM = 10
        self.MAX_WEIGHT_ITEM = 15
        self.MIN_VALUE_ITEM = 10
        self.MAX_VALUE_ITEM = 15
        self.INITIAL_TASK_DISTRIBUTION_FIXED = True
        self.MAX_TRAINING_STEPS = self.NUM_ITEM * 2_000
        self.BUFFER_CAPACITY = self.NUM_ITEM * 1_000


class ConfigKnapsack0Ppo(ConfigBase, ConfigKnapsack0, ConfigPpo):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigKnapsack0.__init__(self)
        ConfigPpo.__init__(self)

        self.NUM_ITEM = 20
        self.LIMIT_WEIGHT_KNAPSACK = 200
        self.MIN_WEIGHT_ITEM = 10
        self.MAX_WEIGHT_ITEM = 15
        self.MIN_VALUE_ITEM = 10
        self.MAX_VALUE_ITEM = 15
        self.INITIAL_TASK_DISTRIBUTION_FIXED = True
        self.MAX_TRAINING_STEPS = self.NUM_ITEM * 2_000
        self.BUFFER_CAPACITY = self.NUM_ITEM * 1_000
