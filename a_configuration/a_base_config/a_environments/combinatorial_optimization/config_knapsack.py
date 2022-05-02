from g_utils.types import ModelType

class ConfigKnapsack:
    def __init__(self):
        self.EPISODE_REWARD_AVG_SOLVED = 100
        self.EPISODE_REWARD_STD_SOLVED = 20

        self.CONSOLE_LOG_INTERVAL_TRAINING_STEPS = 50
        self.MODEL_TYPE = ModelType.TINY_1D_CONVOLUTIONAL
        self.TARGET_SYNC_INTERVAL_TRAINING_STEPS = 100

        self.NUM_ITEM = 20

        self.MIN_WEIGHT_ITEM = 10
        self.MAX_WEIGHT_ITEM = 15

        self.MIN_VALUE_ITEM = 10
        self.MAX_VALUE_ITEM = 15

        self.INITIAL_ITEM_DISTRIBUTION_FIXED = True

        self.SOLUTION_FOUND = [0]

        self.MAX_TRAINING_STEPS = self.NUM_ITEM * 2_000
        self.BUFFER_CAPACITY = self.NUM_ITEM * 1_000

        self.FILE_PATH = False
        self.UPLOAD_PATH = False
        self.OPTIMAL_PATH = False
        self.INSTANCE_INDEX = False

        self.SORTING_TYPE = None
        self.ENV_NAME = "Knapsack_Problem_v0"
        self.LEARNING_RATE = 0.0003

        self.STATIC_INITIAL_STATE_50 = False


class ConfigKnapsack0(ConfigKnapsack):
    def __init__(self):
        ConfigKnapsack.__init__(self)

        self.LIMIT_WEIGHT_KNAPSACK = 200

        self.FILE_PATH = 'knapsack_instances/RI/instances/n_50_r_100/instance0.csv'
        self.UPLOAD_PATH = 'knapsack_instances/RI/link_solution/n_50_r_100/link_solution0.csv'
        self.OPTIMAL_PATH = 'knapsack_instances/RI/optimal_solution/n_50_r_100/solution0.csv'
        self.INSTANCE_INDEX = 0


class ConfigKnapsackTest(ConfigKnapsack):
    def __init__(self):
        ConfigKnapsack.__init__(self)


class ConfigKnapsackStaticTest(ConfigKnapsack):
    def __init__(self):
        ConfigKnapsack.__init__(self)

        self.STATIC_INITIAL_STATE_50 = True

