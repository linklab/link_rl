from a_configuration.a_base_config.b_agents.config_agents_off_policy import ConfigDqn
from a_configuration.a_base_config.config_single_base import ConfigBase
from a_configuration.a_base_config.a_environments.config_knapsack_problem import ConfigKnapsackProblem0


class ConfigKnapsackProblem0Dqn(ConfigBase, ConfigKnapsackProblem0, ConfigDqn):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigKnapsackProblem0.__init__(self)
        ConfigDqn.__init__(self)

        self.NUM_ITEM = 20
        self.MAX_WEIGHT = 200
        self.MIN_WEIGHT_ITEM = 10
        self.MAX_WEIGHT_ITEM = 15
        self.MIN_VALUE_ITEM = 10
        self.MAX_VALUE_ITEM = 15
        self.INITIAL_TASK_DISTRIBUTION_FIXED = True
        self.MAX_TRAINING_STEPS = self.NUM_ITEM * 2_000
        self.BUFFER_CAPACITY = self.NUM_ITEM * 1_000
