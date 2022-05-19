from a_configuration.a_base_config.a_environments.combinatorial_optimization.knapsack.config_new_knapsack import \
    ConfigNewKnapsack0StaticTestLinear
from a_configuration.a_base_config.b_agents.config_agents_off_policy import ConfigDoubleDqn
from a_configuration.a_base_config.config_single_base import ConfigBase


class ConfigNewKnapsack0StaticTestLinearDoubleDqn(
    ConfigBase, ConfigNewKnapsack0StaticTestLinear, ConfigDoubleDqn
):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigNewKnapsack0StaticTestLinear.__init__(self)
        ConfigDoubleDqn.__init__(self)

        self.NUM_ITEM = 50
        self.LIMIT_WEIGHT_KNAPSACK = 200

        self.SORTING_TYPE = None

        self.INITIAL_ITEM_DISTRIBUTION_FIXED = True

        self.MAX_TRAINING_STEPS = self.NUM_ITEM * 20_000
        self.BUFFER_CAPACITY = self.NUM_ITEM * 2_000
