from g_utils.types import ModelType


class ConfigKnapsack:
    def __init__(self):
        self.ENV_NAME = "Knapsack_Problem_v0"

        self.NUM_ITEM = 20
        self.LIMIT_WEIGHT_KNAPSACK = 200

        self.MIN_WEIGHT_ITEM = 10
        self.MAX_WEIGHT_ITEM = 15

        self.MIN_VALUE_ITEM = 10
        self.MAX_VALUE_ITEM = 15

        self.INITIAL_ITEM_DISTRIBUTION_FIXED = True

        self.SORTING_TYPE = None

        self.INITIAL_STATE_FILE_PATH = False
        self.UPLOAD_PATH = False
        self.OPTIMAL_PATH = False
        self.INSTANCE_INDEX = False

        self.STATIC_INITIAL_STATE_50 = False

        self.LEARNING_RATE = 0.0003

        self.MAX_TRAINING_STEPS = self.NUM_ITEM * 2_000
        self.BUFFER_CAPACITY = self.NUM_ITEM * 2_000

        self.EPISODE_REWARD_AVG_SOLVED = 100
        self.EPISODE_REWARD_STD_SOLVED = 20

        self.CONSOLE_LOG_INTERVAL_TRAINING_STEPS = 50
        self.MODEL_TYPE = ModelType.TINY_1D_CONVOLUTIONAL
        self.TARGET_SYNC_INTERVAL_TRAINING_STEPS = 100


class ConfigKnapsack0RandomTest(ConfigKnapsack):
    def __init__(self):
        ConfigKnapsack.__init__(self)


class ConfigKnapsack0RandomTestLinear(ConfigKnapsack):
    def __init__(self):
        ConfigKnapsack.__init__(self)

        self.MODEL_TYPE = ModelType.MEDIUM_LINEAR


class ConfigKnapsack0LoadTest(ConfigKnapsack):
    def __init__(self):
        ConfigKnapsack.__init__(self)

        self.NUM_ITEM = 50
        self.INITIAL_STATE_FILE_PATH = 'knapsack_instances/RI/instances/n_50_r_100'
        self.UPLOAD_PATH = 'knapsack_instances/RI/link_solution/n_50_r_100'
        self.OPTIMAL_PATH = 'knapsack_instances/RI/optimal_solution/n_50_r_100'
        self.INSTANCE_INDEX = 0


class ConfigKnapsack0LoadTestLinear(ConfigKnapsack):
    def __init__(self):
        ConfigKnapsack.__init__(self)

        self.NUM_ITEM = 50
        self.INITIAL_STATE_FILE_PATH = 'knapsack_instances/RI/instances/n_50_r_100'
        self.UPLOAD_PATH = 'knapsack_instances/RI/link_solution/n_50_r_100'
        self.OPTIMAL_PATH = 'knapsack_instances/RI/optimal_solution/n_50_r_100'
        self.INSTANCE_INDEX = 0

        self.MODEL_TYPE = ModelType.MEDIUM_LINEAR


class ConfigKnapsack0StaticTest(ConfigKnapsack):
    def __init__(self):
        ConfigKnapsack.__init__(self)

        self.NUM_ITEM = 50
        self.STATIC_INITIAL_STATE_50 = True


class ConfigKnapsack0StaticTestLinear(ConfigKnapsack):
    def __init__(self):
        ConfigKnapsack.__init__(self)

        self.NUM_ITEM = 50
        self.STATIC_INITIAL_STATE_50 = True
        self.MODEL_TYPE = ModelType.MEDIUM_LINEAR


class ConfigKnapsack0StaticTestLinearRecurrent(ConfigKnapsack):
    def __init__(self):
        ConfigKnapsack.__init__(self)

        self.NUM_ITEM = 50
        self.STATIC_INITIAL_STATE_50 = True
        self.MODEL_TYPE = ModelType.MEDIUM_RECURRENT


##################################
##Action_Space = NUMBER_OF_ITEMS##
##################################

class ConfigKnapsack1(ConfigKnapsack):
    def __init__(self):
        ConfigKnapsack.__init__(self)

        self.ENV_NAME = "Knapsack_Problem_v1"


class ConfigKnapsack1RandomTest(ConfigKnapsack1):
    def __init__(self):
        ConfigKnapsack1.__init__(self)


class ConfigKnapsack1RandomTestLinear(ConfigKnapsack1):
    def __init__(self):
        ConfigKnapsack1.__init__(self)

        self.MODEL_TYPE = ModelType.MEDIUM_LINEAR


class ConfigKnapsack1LoadTest(ConfigKnapsack1):
    def __init__(self):
        ConfigKnapsack1.__init__(self)

        self.NUM_ITEM = 50
        self.INITIAL_STATE_FILE_PATH = 'knapsack_instances/RI/instances/n_50_r_100'
        self.UPLOAD_PATH = 'knapsack_instances/RI/link_solution/n_50_r_100'
        self.OPTIMAL_PATH = 'knapsack_instances/RI/optimal_solution/n_50_r_100'
        self.INSTANCE_INDEX = 0


class ConfigKnapsack1LoadTestLinear(ConfigKnapsack1):
    def __init__(self):
        ConfigKnapsack1.__init__(self)

        self.NUM_ITEM = 50
        self.INITIAL_STATE_FILE_PATH = 'knapsack_instances/RI/instances/n_50_r_100'
        self.UPLOAD_PATH = 'knapsack_instances/RI/link_solution/n_50_r_100'
        self.OPTIMAL_PATH = 'knapsack_instances/RI/optimal_solution/n_50_r_100'
        self.INSTANCE_INDEX = 0

        self.MODEL_TYPE = ModelType.MEDIUM_LINEAR


class ConfigKnapsack1StaticTest(ConfigKnapsack1):
    def __init__(self):
        ConfigKnapsack1.__init__(self)

        self.NUM_ITEM = 50
        self.STATIC_INITIAL_STATE_50 = True


class ConfigKnapsack1StaticTestLinear(ConfigKnapsack1):
    def __init__(self):
        ConfigKnapsack1.__init__(self)

        self.NUM_ITEM = 50
        self.STATIC_INITIAL_STATE_50 = True
        self.MODEL_TYPE = ModelType.MEDIUM_LINEAR
