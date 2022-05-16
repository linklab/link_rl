import os
import random

from a_configuration.a_base_config.a_environments.combinatorial_optimization.config_knapsack import \
    ConfigKnapsack0RandomTest, ConfigKnapsack0RandomTestLinear, ConfigKnapsack0LoadTest, ConfigKnapsack0LoadTestLinear, \
    ConfigKnapsack0StaticTest, ConfigKnapsack0StaticTestLinear, ConfigKnapsack1RandomTest, \
    ConfigKnapsack1RandomTestLinear, ConfigKnapsack1LoadTest, ConfigKnapsack1LoadTestLinear, ConfigKnapsack1StaticTest, \
    ConfigKnapsack1StaticTestLinear
from a_configuration.a_base_config.b_agents.config_agents_off_policy import ConfigDqn
from a_configuration.a_base_config.b_agents.config_agents_on_policy import ConfigA2c, ConfigPpo
from a_configuration.a_base_config.config_single_base import ConfigBase

#####################################
######### Agent_Type = DQN ##########
#####################################

class ConfigKnapsack0RandomTestDqn(ConfigBase, ConfigKnapsack0RandomTest, ConfigDqn):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigKnapsack0RandomTest.__init__(self)
        ConfigDqn.__init__(self)

        self.NUM_ITEM = 20
        self.LIMIT_WEIGHT_KNAPSACK = 200
        self.MIN_WEIGHT_ITEM = 10
        self.MAX_WEIGHT_ITEM = 15
        self.MIN_VALUE_ITEM = 10
        self.MAX_VALUE_ITEM = 15

        self.SORTING_TYPE = None

        self.INITIAL_ITEM_DISTRIBUTION_FIXED = True

        self.MAX_TRAINING_STEPS = self.NUM_ITEM * 2_000
        self.BUFFER_CAPACITY = self.NUM_ITEM * 1_000


class ConfigKnapsack0RandomTestLinearDqn(ConfigBase, ConfigKnapsack0RandomTestLinear, ConfigDqn):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigKnapsack0RandomTestLinear.__init__(self)
        ConfigDqn.__init__(self)

        self.NUM_ITEM = 20
        self.LIMIT_WEIGHT_KNAPSACK = 200
        self.MIN_WEIGHT_ITEM = 10
        self.MAX_WEIGHT_ITEM = 15
        self.MIN_VALUE_ITEM = 10
        self.MAX_VALUE_ITEM = 15

        self.SORTING_TYPE = None

        self.INITIAL_ITEM_DISTRIBUTION_FIXED = True

        self.MAX_TRAINING_STEPS = self.NUM_ITEM * 2_000
        self.BUFFER_CAPACITY = self.NUM_ITEM * 1_000


class ConfigKnapsack0LoadTestDqn(ConfigBase, ConfigKnapsack0LoadTest, ConfigDqn):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigKnapsack0LoadTest.__init__(self)
        ConfigDqn.__init__(self)

        self.NUM_ITEM = 50

        self.FILE_PATH = 'knapsack_instances/RI/instances/n_50_r_100'
        self.UPLOAD_PATH = 'knapsack_instances/RI/link_solution/n_50_r_100'
        self.OPTIMAL_PATH = 'knapsack_instances/RI/optimal_solution/n_50_r_100'

        self.SORTING_TYPE = None

        self.INITIAL_ITEM_DISTRIBUTION_FIXED = True

        self.MAX_TRAINING_STEPS = self.NUM_ITEM * 3_000
        self.BUFFER_CAPACITY = self.NUM_ITEM * 2_000


class ConfigKnapsack0LoadTestLinearDqn(ConfigBase, ConfigKnapsack0LoadTestLinear, ConfigDqn):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigKnapsack0LoadTestLinear.__init__(self)
        ConfigDqn.__init__(self)

        self.NUM_ITEM = 50

        self.FILE_PATH = 'knapsack_instances/RI/instances/n_50_r_100'
        self.UPLOAD_PATH = 'knapsack_instances/RI/link_solution/n_50_r_100'
        self.OPTIMAL_PATH = 'knapsack_instances/RI/optimal_solution/n_50_r_100'

        self.SORTING_TYPE = None

        self.INITIAL_ITEM_DISTRIBUTION_FIXED = True

        self.MAX_TRAINING_STEPS = self.NUM_ITEM * 3_000
        self.BUFFER_CAPACITY = self.NUM_ITEM * 2_000


class ConfigKnapsack0StaticTestDqn(ConfigBase, ConfigKnapsack0StaticTest, ConfigDqn):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigKnapsack0StaticTest.__init__(self)
        ConfigDqn.__init__(self)

        self.NUM_ITEM = 50
        self.LIMIT_WEIGHT_KNAPSACK = 200

        self.SORTING_TYPE = None

        self.INITIAL_ITEM_DISTRIBUTION_FIXED = True

        self.MAX_TRAINING_STEPS = self.NUM_ITEM * 20_000
        self.BUFFER_CAPACITY = self.NUM_ITEM * 1_000


class ConfigKnapsack0StaticTestLinearDqn(ConfigBase, ConfigKnapsack0StaticTestLinear, ConfigDqn):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigKnapsack0StaticTestLinear.__init__(self)
        ConfigDqn.__init__(self)

        self.NUM_ITEM = 50
        self.LIMIT_WEIGHT_KNAPSACK = 200

        self.SORTING_TYPE = None

        self.INITIAL_ITEM_DISTRIBUTION_FIXED = True

        self.MAX_TRAINING_STEPS = self.NUM_ITEM * 20_000
        self.BUFFER_CAPACITY = self.NUM_ITEM * 1_000


#####################################
######### Agent_Type = A2C ##########
#####################################
class ConfigKnapsack0RandomTestLinearA2c(ConfigBase, ConfigKnapsack0RandomTestLinear, ConfigA2c):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigKnapsack0RandomTestLinear.__init__(self)
        ConfigA2c.__init__(self)

        self.NUM_ITEM = 20
        self.LIMIT_WEIGHT_KNAPSACK = 200
        self.MIN_WEIGHT_ITEM = 10
        self.MAX_WEIGHT_ITEM = 15
        self.MIN_VALUE_ITEM = 10
        self.MAX_VALUE_ITEM = 15

        self.SORTING_TYPE = None

        self.INITIAL_ITEM_DISTRIBUTION_FIXED = True

        self.MAX_TRAINING_STEPS = self.NUM_ITEM * 2_000
        self.BUFFER_CAPACITY = self.NUM_ITEM * 1_000


class ConfigKnapsack0LoadTestLinearA2c(ConfigBase, ConfigKnapsack0LoadTestLinear, ConfigA2c):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigKnapsack0LoadTestLinear.__init__(self)
        ConfigA2c.__init__(self)

        self.NUM_ITEM = 50

        self.FILE_PATH = 'knapsack_instances/RI/instances/n_50_r_100'
        self.UPLOAD_PATH = 'knapsack_instances/RI/link_solution/n_50_r_100'
        self.OPTIMAL_PATH = 'knapsack_instances/RI/optimal_solution/n_50_r_100'

        self.SORTING_TYPE = None

        self.INITIAL_ITEM_DISTRIBUTION_FIXED = True

        self.MAX_TRAINING_STEPS = self.NUM_ITEM * 3_000
        self.BUFFER_CAPACITY = self.NUM_ITEM * 2_000


class ConfigKnapsack0StaticTestLinearA2c(ConfigBase, ConfigKnapsack0StaticTestLinear, ConfigA2c):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigKnapsack0StaticTestLinear.__init__(self)
        ConfigA2c.__init__(self)

        self.NUM_ITEM = 50
        self.LIMIT_WEIGHT_KNAPSACK = 200

        self.SORTING_TYPE = None

        self.INITIAL_ITEM_DISTRIBUTION_FIXED = True

        self.MAX_TRAINING_STEPS = self.NUM_ITEM * 20_000
        self.BUFFER_CAPACITY = self.NUM_ITEM * 1_000


#####################################
######### Agent_Type = Ppo ##########
#####################################
class ConfigKnapsack0RandomTestLinearPpo(ConfigBase, ConfigKnapsack0RandomTestLinear, ConfigPpo):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigKnapsack0RandomTestLinear.__init__(self)
        ConfigPpo.__init__(self)

        self.NUM_ITEM = 20
        self.LIMIT_WEIGHT_KNAPSACK = 200
        self.MIN_WEIGHT_ITEM = 10
        self.MAX_WEIGHT_ITEM = 15
        self.MIN_VALUE_ITEM = 10
        self.MAX_VALUE_ITEM = 15

        self.SORTING_TYPE = None

        self.INITIAL_ITEM_DISTRIBUTION_FIXED = True

        self.MAX_TRAINING_STEPS = self.NUM_ITEM * 2_000
        self.BUFFER_CAPACITY = self.NUM_ITEM * 1_000


class ConfigKnapsack0LoadTestLinearPpo(ConfigBase, ConfigKnapsack0LoadTestLinear, ConfigPpo):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigKnapsack0LoadTestLinear.__init__(self)
        ConfigPpo.__init__(self)

        self.NUM_ITEM = 50

        self.FILE_PATH = 'knapsack_instances/RI/instances/n_50_r_100'
        self.UPLOAD_PATH = 'knapsack_instances/RI/link_solution/n_50_r_100'
        self.OPTIMAL_PATH = 'knapsack_instances/RI/optimal_solution/n_50_r_100'

        self.SORTING_TYPE = None

        self.INITIAL_ITEM_DISTRIBUTION_FIXED = True

        self.MAX_TRAINING_STEPS = self.NUM_ITEM * 3_000
        self.BUFFER_CAPACITY = self.NUM_ITEM * 2_000


class ConfigKnapsack0StaticTestLinearPpo(ConfigBase, ConfigKnapsack0StaticTestLinear, ConfigPpo):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigKnapsack0StaticTestLinear.__init__(self)
        ConfigPpo.__init__(self)

        self.NUM_ITEM = 50
        self.LIMIT_WEIGHT_KNAPSACK = 200

        self.SORTING_TYPE = None

        self.INITIAL_ITEM_DISTRIBUTION_FIXED = True

        self.MAX_TRAINING_STEPS = self.NUM_ITEM * 20_000
        self.BUFFER_CAPACITY = self.NUM_ITEM * 1_000


#####################################
####Action_Space = NUMBER_OF_ITEMS###
######### Agent_Type = DQN ##########
#####################################

class ConfigKnapsack1RandomTestDqn(ConfigBase, ConfigKnapsack1RandomTest, ConfigDqn):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigKnapsack1RandomTest.__init__(self)
        ConfigDqn.__init__(self)

        self.NUM_ITEM = 20
        self.LIMIT_WEIGHT_KNAPSACK = 200
        self.MIN_WEIGHT_ITEM = 10
        self.MAX_WEIGHT_ITEM = 15
        self.MIN_VALUE_ITEM = 10
        self.MAX_VALUE_ITEM = 15

        self.SORTING_TYPE = None

        self.INITIAL_ITEM_DISTRIBUTION_FIXED = True

        self.MAX_TRAINING_STEPS = self.NUM_ITEM * 2_000
        self.BUFFER_CAPACITY = self.NUM_ITEM * 1_000


class ConfigKnapsack1RandomTestLinearDqn(ConfigBase, ConfigKnapsack1RandomTestLinear, ConfigDqn):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigKnapsack1RandomTestLinear.__init__(self)
        ConfigDqn.__init__(self)

        self.NUM_ITEM = 20
        self.LIMIT_WEIGHT_KNAPSACK = 200
        self.MIN_WEIGHT_ITEM = 10
        self.MAX_WEIGHT_ITEM = 15
        self.MIN_VALUE_ITEM = 10
        self.MAX_VALUE_ITEM = 15

        self.SORTING_TYPE = None

        self.INITIAL_ITEM_DISTRIBUTION_FIXED = True

        self.MAX_TRAINING_STEPS = self.NUM_ITEM * 2_000
        self.BUFFER_CAPACITY = self.NUM_ITEM * 1_000


class ConfigKnapsack1LoadTestDqn(ConfigBase, ConfigKnapsack1LoadTest, ConfigDqn):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigKnapsack1LoadTest.__init__(self)
        ConfigDqn.__init__(self)

        self.NUM_ITEM = 50

        self.FILE_PATH = 'knapsack_instances/RI/instances/n_50_r_100'
        self.UPLOAD_PATH = 'knapsack_instances/RI/link_solution/n_50_r_100'
        self.OPTIMAL_PATH = 'knapsack_instances/RI/optimal_solution/n_50_r_100'

        self.SORTING_TYPE = None

        self.INITIAL_ITEM_DISTRIBUTION_FIXED = True

        self.MAX_TRAINING_STEPS = self.NUM_ITEM * 3_000
        self.BUFFER_CAPACITY = self.NUM_ITEM * 2_000


class ConfigKnapsack1LoadTestLinearDqn(ConfigBase, ConfigKnapsack1LoadTestLinear, ConfigDqn):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigKnapsack1LoadTestLinear.__init__(self)
        ConfigDqn.__init__(self)

        self.NUM_ITEM = 50

        self.FILE_PATH = 'knapsack_instances/RI/instances/n_50_r_100'
        self.UPLOAD_PATH = 'knapsack_instances/RI/link_solution/n_50_r_100'
        self.OPTIMAL_PATH = 'knapsack_instances/RI/optimal_solution/n_50_r_100'

        self.SORTING_TYPE = None

        self.INITIAL_ITEM_DISTRIBUTION_FIXED = True

        self.MAX_TRAINING_STEPS = self.NUM_ITEM * 3_000
        self.BUFFER_CAPACITY = self.NUM_ITEM * 2_000


class ConfigKnapsack1StaticTestDqn(ConfigBase, ConfigKnapsack1StaticTest, ConfigDqn):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigKnapsack1StaticTest.__init__(self)
        ConfigDqn.__init__(self)

        self.NUM_ITEM = 50
        self.LIMIT_WEIGHT_KNAPSACK = 200

        self.SORTING_TYPE = None

        self.INITIAL_ITEM_DISTRIBUTION_FIXED = True

        self.MAX_TRAINING_STEPS = self.NUM_ITEM * 20_000
        self.BUFFER_CAPACITY = self.NUM_ITEM * 1_000


class ConfigKnapsack1StaticTestLinearDqn(ConfigBase, ConfigKnapsack1StaticTestLinear, ConfigDqn):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigKnapsack1StaticTestLinear.__init__(self)
        ConfigDqn.__init__(self)

        self.NUM_ITEM = 50
        self.LIMIT_WEIGHT_KNAPSACK = 200

        self.SORTING_TYPE = None

        self.INITIAL_ITEM_DISTRIBUTION_FIXED = True

        self.MAX_TRAINING_STEPS = self.NUM_ITEM * 20_000
        self.BUFFER_CAPACITY = self.NUM_ITEM * 1_000


#####################################
######### Agent_Type = A2C ##########
#####################################
class ConfigKnapsack1RandomTestLinearA2c(ConfigBase, ConfigKnapsack1RandomTestLinear, ConfigA2c):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigKnapsack1RandomTestLinear.__init__(self)
        ConfigA2c.__init__(self)

        self.NUM_ITEM = 20
        self.LIMIT_WEIGHT_KNAPSACK = 200
        self.MIN_WEIGHT_ITEM = 10
        self.MAX_WEIGHT_ITEM = 15
        self.MIN_VALUE_ITEM = 10
        self.MAX_VALUE_ITEM = 15

        self.SORTING_TYPE = None

        self.INITIAL_ITEM_DISTRIBUTION_FIXED = True

        self.MAX_TRAINING_STEPS = self.NUM_ITEM * 2_000
        self.BUFFER_CAPACITY = self.NUM_ITEM * 1_000


class ConfigKnapsack1LoadTestLinearA2c(ConfigBase, ConfigKnapsack1LoadTestLinear, ConfigA2c):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigKnapsack1LoadTestLinear.__init__(self)
        ConfigA2c.__init__(self)

        self.NUM_ITEM = 50

        self.FILE_PATH = 'knapsack_instances/RI/instances/n_50_r_100'
        self.UPLOAD_PATH = 'knapsack_instances/RI/link_solution/n_50_r_100'
        self.OPTIMAL_PATH = 'knapsack_instances/RI/optimal_solution/n_50_r_100'

        self.SORTING_TYPE = None

        self.INITIAL_ITEM_DISTRIBUTION_FIXED = True

        self.MAX_TRAINING_STEPS = self.NUM_ITEM * 3_000
        self.BUFFER_CAPACITY = self.NUM_ITEM * 2_000


class ConfigKnapsack1StaticTestLinearA2c(ConfigBase, ConfigKnapsack1StaticTestLinear, ConfigA2c):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigKnapsack1StaticTestLinear.__init__(self)
        ConfigA2c.__init__(self)

        self.NUM_ITEM = 50
        self.LIMIT_WEIGHT_KNAPSACK = 200

        self.SORTING_TYPE = None

        self.INITIAL_ITEM_DISTRIBUTION_FIXED = True

        self.MAX_TRAINING_STEPS = self.NUM_ITEM * 20_000
        self.BUFFER_CAPACITY = self.NUM_ITEM * 1_000


#####################################
######### Agent_Type = Ppo ##########
#####################################
class ConfigKnapsack1RandomTestLinearPpo(ConfigBase, ConfigKnapsack1RandomTestLinear, ConfigPpo):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigKnapsack1RandomTestLinear.__init__(self)
        ConfigPpo.__init__(self)

        self.NUM_ITEM = 20
        self.LIMIT_WEIGHT_KNAPSACK = 200
        self.MIN_WEIGHT_ITEM = 10
        self.MAX_WEIGHT_ITEM = 15
        self.MIN_VALUE_ITEM = 10
        self.MAX_VALUE_ITEM = 15

        self.SORTING_TYPE = None

        self.INITIAL_ITEM_DISTRIBUTION_FIXED = True

        self.MAX_TRAINING_STEPS = self.NUM_ITEM * 2_000
        self.BUFFER_CAPACITY = self.NUM_ITEM * 1_000


class ConfigKnapsack1LoadTestLinearPpo(ConfigBase, ConfigKnapsack1LoadTestLinear, ConfigPpo):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigKnapsack1LoadTestLinear.__init__(self)
        ConfigPpo.__init__(self)

        self.NUM_ITEM = 50

        self.FILE_PATH = 'knapsack_instances/RI/instances/n_50_r_100'
        self.UPLOAD_PATH = 'knapsack_instances/RI/link_solution/n_50_r_100'
        self.OPTIMAL_PATH = 'knapsack_instances/RI/optimal_solution/n_50_r_100'

        self.SORTING_TYPE = None

        self.INITIAL_ITEM_DISTRIBUTION_FIXED = True

        self.MAX_TRAINING_STEPS = self.NUM_ITEM * 3_000
        self.BUFFER_CAPACITY = self.NUM_ITEM * 2_000


class ConfigKnapsack1StaticTestLinearPpo(ConfigBase, ConfigKnapsack1StaticTestLinear, ConfigPpo):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigKnapsack1StaticTestLinear.__init__(self)
        ConfigPpo.__init__(self)

        self.NUM_ITEM = 50
        self.LIMIT_WEIGHT_KNAPSACK = 200

        self.SORTING_TYPE = None

        self.INITIAL_ITEM_DISTRIBUTION_FIXED = True

        self.MAX_TRAINING_STEPS = self.NUM_ITEM * 20_000
        self.BUFFER_CAPACITY = self.NUM_ITEM * 1_000