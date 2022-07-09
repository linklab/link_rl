from link_rl.a_configuration.a_base_config.a_environments.combinatorial_optimization.knapsack.config_knapsack import \
    ConfigKnapsack0RandomTest, ConfigKnapsack0RandomTestLinear, ConfigKnapsack0LoadTest, ConfigKnapsack0LoadTestLinear, \
    ConfigKnapsack0StaticTest, ConfigKnapsack0StaticTestLinear
from link_rl.a_configuration.a_base_config.b_agents.config_agents_off_policy import ConfigDqn, ConfigDoubleDuelingDqn, \
    ConfigDoubleDqn
from link_rl.a_configuration.a_base_config.b_agents.config_agents_on_policy import ConfigA2c, ConfigPpo
from link_rl.a_configuration.a_base_config.config_single_base import ConfigBase

#####################################
######### Agent_Type = DQN ##########
#####################################
from link_rl.c_models_v2.b_q_model import Q_MODEL
from link_rl.c_models_v2.d_basic_actor_critic_model import BASIC_ACTOR_CRITIC_MODEL


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

        self.SORTING_TYPE = 1

        self.INITIAL_ITEM_DISTRIBUTION_FIXED = False

        self.MAX_TRAINING_STEPS = self.NUM_ITEM * 50_000
        self.BUFFER_CAPACITY = self.NUM_ITEM * 10_000

        self.GAMMA = 0.999
        self.LEARNING_RATE = 0.001
        self.MODEL_TYPE = Q_MODEL.QModel.value


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

        self.SORTING_TYPE = 1

        self.INITIAL_ITEM_DISTRIBUTION_FIXED = False

        self.MAX_TRAINING_STEPS = self.NUM_ITEM * 50_000
        self.BUFFER_CAPACITY = self.NUM_ITEM * 10_000

        self.GAMMA = 0.999
        self.LEARNING_RATE = 0.001
        self.MODEL_TYPE = Q_MODEL.QModel.value


class ConfigKnapsack0RandomTestLinearDoubleDqn(ConfigBase, ConfigKnapsack0RandomTestLinear, ConfigDoubleDqn):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigKnapsack0RandomTestLinear.__init__(self)
        ConfigDoubleDqn.__init__(self)

        self.NUM_ITEM = 20
        self.LIMIT_WEIGHT_KNAPSACK = 200
        self.MIN_WEIGHT_ITEM = 10
        self.MAX_WEIGHT_ITEM = 15
        self.MIN_VALUE_ITEM = 10
        self.MAX_VALUE_ITEM = 15

        self.SORTING_TYPE = 1

        self.INITIAL_ITEM_DISTRIBUTION_FIXED = False

        self.MAX_TRAINING_STEPS = self.NUM_ITEM * 50_000
        self.BUFFER_CAPACITY = self.NUM_ITEM * 10_000

        self.GAMMA = 0.999
        self.LEARNING_RATE = 0.001
        self.MODEL_TYPE = Q_MODEL.QModel.value
        

class ConfigKnapsack0LoadTestDqn(ConfigBase, ConfigKnapsack0LoadTest, ConfigDqn):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigKnapsack0LoadTest.__init__(self)
        ConfigDqn.__init__(self)

        self.NUM_ITEM = 50

        self.INITIAL_STATE_FILE_PATH = 'knapsack_instances/RI/instances/n_50_r_100'
        self.UPLOAD_PATH = 'knapsack_instances/RI/link_solution/n_50_r_100'
        self.OPTIMAL_PATH = 'knapsack_instances/RI/optimal_solution/n_50_r_100'

        self.SORTING_TYPE = 1

        self.INITIAL_ITEM_DISTRIBUTION_FIXED = True

        self.MAX_TRAINING_STEPS = self.NUM_ITEM * 3_000
        self.BUFFER_CAPACITY = self.NUM_ITEM * 2_000

        self.GAMMA = 0.999
        self.LEARNING_RATE = 0.001
        self.MODEL_TYPE = Q_MODEL.QModel.value


class ConfigKnapsack0LoadTestLinearDqn(ConfigBase, ConfigKnapsack0LoadTestLinear, ConfigDqn):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigKnapsack0LoadTestLinear.__init__(self)
        ConfigDqn.__init__(self)

        self.NUM_ITEM = 50

        self.INITIAL_STATE_FILE_PATH = 'knapsack_instances/RI/instances/n_50_r_100'
        self.UPLOAD_PATH = 'knapsack_instances/RI/link_solution/n_50_r_100'
        self.OPTIMAL_PATH = 'knapsack_instances/RI/optimal_solution/n_50_r_100'

        self.SORTING_TYPE = 1

        self.INITIAL_ITEM_DISTRIBUTION_FIXED = True

        self.MAX_TRAINING_STEPS = self.NUM_ITEM * 3_000
        self.BUFFER_CAPACITY = self.NUM_ITEM * 2_000

        self.GAMMA = 0.999
        self.LEARNING_RATE = 0.001
        self.MODEL_TYPE = Q_MODEL.QModel.value


class ConfigKnapsack0LoadTestLinearDoubleDqn(ConfigBase, ConfigKnapsack0LoadTestLinear, ConfigDoubleDqn):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigKnapsack0LoadTestLinear.__init__(self)
        ConfigDoubleDqn.__init__(self)

        self.NUM_ITEM = 50
        self.LIMIT_WEIGHT_KNAPSACK = 200

        self.SORTING_TYPE = 1

        self.INITIAL_ITEM_DISTRIBUTION_FIXED = True

        self.MAX_TRAINING_STEPS = self.NUM_ITEM * 20_000
        self.BUFFER_CAPACITY = self.NUM_ITEM * 2_000

        self.GAMMA = 0.999
        self.LEARNING_RATE = 0.001
        self.MODEL_TYPE = Q_MODEL.QModel.value


class ConfigKnapsack0StaticTestDqn(ConfigBase, ConfigKnapsack0StaticTest, ConfigDqn):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigKnapsack0StaticTest.__init__(self)
        ConfigDqn.__init__(self)

        self.NUM_ITEM = 50
        self.LIMIT_WEIGHT_KNAPSACK = 200

        self.SORTING_TYPE = 1

        self.INITIAL_ITEM_DISTRIBUTION_FIXED = True

        self.MAX_TRAINING_STEPS = self.NUM_ITEM * 20_000
        self.BUFFER_CAPACITY = self.NUM_ITEM * 2_000

        self.GAMMA = 0.999
        self.LEARNING_RATE = 0.001
        self.MODEL_TYPE = Q_MODEL.QModel.value


class ConfigKnapsack0StaticTestLinearDqn(ConfigBase, ConfigKnapsack0StaticTestLinear, ConfigDqn):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigKnapsack0StaticTestLinear.__init__(self)
        ConfigDqn.__init__(self)

        self.NUM_ITEM = 50
        self.LIMIT_WEIGHT_KNAPSACK = 200

        self.SORTING_TYPE = 1

        self.INITIAL_ITEM_DISTRIBUTION_FIXED = True

        self.MAX_TRAINING_STEPS = self.NUM_ITEM * 20_000
        self.BUFFER_CAPACITY = self.NUM_ITEM * 2_000

        self.GAMMA = 0.999
        self.LEARNING_RATE = 0.001
        self.MODEL_TYPE = Q_MODEL.QModel.value


class ConfigKnapsack0StaticTestLinearDoubleDqn(
    ConfigBase, ConfigKnapsack0StaticTestLinear, ConfigDoubleDqn
):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigKnapsack0StaticTestLinear.__init__(self)
        ConfigDoubleDqn.__init__(self)

        self.NUM_ITEM = 50
        self.LIMIT_WEIGHT_KNAPSACK = 200

        self.SORTING_TYPE = 1

        self.INITIAL_ITEM_DISTRIBUTION_FIXED = True

        self.MAX_TRAINING_STEPS = self.NUM_ITEM * 2_000
        self.BUFFER_CAPACITY = self.NUM_ITEM * 2_000

        self.GAMMA = 0.999
        self.LEARNING_RATE = 0.001
        self.MODEL_TYPE = Q_MODEL.QModel.value


class ConfigKnapsack0StaticTestLinearDoubleDuelingDqn(
    ConfigBase, ConfigKnapsack0StaticTestLinear, ConfigDoubleDuelingDqn
):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigKnapsack0StaticTestLinear.__init__(self)
        ConfigDoubleDuelingDqn.__init__(self)

        self.NUM_ITEM = 50
        self.LIMIT_WEIGHT_KNAPSACK = 200

        self.SORTING_TYPE = 1

        self.INITIAL_ITEM_DISTRIBUTION_FIXED = True

        self.MAX_TRAINING_STEPS = self.NUM_ITEM * 20_000
        self.BUFFER_CAPACITY = self.NUM_ITEM * 2_000

        self.GAMMA = 0.999
        self.LEARNING_RATE = 0.001
        self.MODEL_TYPE = Q_MODEL.QModel.value


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

        self.SORTING_TYPE = 1

        self.INITIAL_ITEM_DISTRIBUTION_FIXED = True

        self.MAX_TRAINING_STEPS = self.NUM_ITEM * 2_000
        self.BUFFER_CAPACITY = self.NUM_ITEM * 2_000

        self.GAMMA = 0.999
        self.LEARNING_RATE = 0.001
        self.MODEL_TYPE = BASIC_ACTOR_CRITIC_MODEL.DiscreteBasicActorCriticSharedModel.value


class ConfigKnapsack0LoadTestLinearA2c(ConfigBase, ConfigKnapsack0LoadTestLinear, ConfigA2c):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigKnapsack0LoadTestLinear.__init__(self)
        ConfigA2c.__init__(self)

        self.NUM_ITEM = 50

        self.INITIAL_STATE_FILE_PATH = 'knapsack_instances/RI/instances/n_50_r_100'
        self.UPLOAD_PATH = 'knapsack_instances/RI/link_solution/n_50_r_100'
        self.OPTIMAL_PATH = 'knapsack_instances/RI/optimal_solution/n_50_r_100'

        self.SORTING_TYPE = 1

        self.INITIAL_ITEM_DISTRIBUTION_FIXED = True

        self.MAX_TRAINING_STEPS = self.NUM_ITEM * 3_000
        self.BUFFER_CAPACITY = self.NUM_ITEM * 2_000

        self.GAMMA = 0.999
        self.LEARNING_RATE = 0.001
        self.MODEL_TYPE = BASIC_ACTOR_CRITIC_MODEL.DiscreteBasicActorCriticSharedModel.value


class ConfigKnapsack0StaticTestLinearA2c(ConfigBase, ConfigKnapsack0StaticTestLinear, ConfigA2c):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigKnapsack0StaticTestLinear.__init__(self)
        ConfigA2c.__init__(self)

        self.NUM_ITEM = 50
        self.LIMIT_WEIGHT_KNAPSACK = 200

        self.SORTING_TYPE = 1

        self.INITIAL_ITEM_DISTRIBUTION_FIXED = True

        self.MAX_TRAINING_STEPS = self.NUM_ITEM * 20_000
        self.BUFFER_CAPACITY = self.NUM_ITEM * 2_000

        self.GAMMA = 0.999
        self.LEARNING_RATE = 0.001
        self.MODEL_TYPE = BASIC_ACTOR_CRITIC_MODEL.DiscreteBasicActorCriticSharedModel.value


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

        self.SORTING_TYPE = 1

        self.INITIAL_ITEM_DISTRIBUTION_FIXED = True

        self.MAX_TRAINING_STEPS = self.NUM_ITEM * 2_000
        self.BUFFER_CAPACITY = self.NUM_ITEM * 2_000

        self.GAMMA = 0.999
        self.LEARNING_RATE = 0.001
        self.MODEL_TYPE = BASIC_ACTOR_CRITIC_MODEL.DiscreteBasicActorCriticSharedModel.value


class ConfigKnapsack0LoadTestLinearPpo(ConfigBase, ConfigKnapsack0LoadTestLinear, ConfigPpo):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigKnapsack0LoadTestLinear.__init__(self)
        ConfigPpo.__init__(self)

        self.NUM_ITEM = 50

        self.INITIAL_STATE_FILE_PATH = 'knapsack_instances/RI/instances/n_50_r_100'
        self.UPLOAD_PATH = 'knapsack_instances/RI/link_solution/n_50_r_100'
        self.OPTIMAL_PATH = 'knapsack_instances/RI/optimal_solution/n_50_r_100'

        self.SORTING_TYPE = 1

        self.INITIAL_ITEM_DISTRIBUTION_FIXED = True

        self.MAX_TRAINING_STEPS = self.NUM_ITEM * 3_000
        self.BUFFER_CAPACITY = self.NUM_ITEM * 2_000

        self.GAMMA = 0.999
        self.LEARNING_RATE = 0.001
        self.MODEL_TYPE = BASIC_ACTOR_CRITIC_MODEL.DiscreteBasicActorCriticSharedModel.value


class ConfigKnapsack0StaticTestLinearPpo(ConfigBase, ConfigKnapsack0StaticTestLinear, ConfigPpo):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigKnapsack0StaticTestLinear.__init__(self)
        ConfigPpo.__init__(self)

        self.NUM_ITEM = 50
        self.LIMIT_WEIGHT_KNAPSACK = 200

        self.SORTING_TYPE = 1

        self.INITIAL_ITEM_DISTRIBUTION_FIXED = True

        self.MAX_TRAINING_STEPS = self.NUM_ITEM * 20_000
        self.BUFFER_CAPACITY = self.NUM_ITEM * 2_000

        self.GAMMA = 0.999
        self.LEARNING_RATE = 0.001
        self.MODEL_TYPE = BASIC_ACTOR_CRITIC_MODEL.DiscreteBasicActorCriticSharedModel.value
