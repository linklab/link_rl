from g_utils.types import ModelType


class ConfigNewKnapsack:
    def __init__(self):
        self.ENV_NAME = "New_Knapsack_Problem_v0"

        self.NUM_ITEM = 20
        self.LIMIT_WEIGHT_KNAPSACK = 200

        self.MIN_WEIGHT_ITEM = 10
        self.MAX_WEIGHT_ITEM = 15

        self.MIN_VALUE_ITEM = 10
        self.MAX_VALUE_ITEM = 15

        self.INITIAL_ITEM_DISTRIBUTION_FIXED = True

        self.SORTING_TYPE = None

        self.FILE_PATH = False
        self.UPLOAD_PATH = False
        self.OPTIMAL_PATH = False
        self.INSTANCE_INDEX = False

        self.SOLUTION_FOUND = [0]

        self.STATIC_INITIAL_STATE_50 = False

        self.LEARNING_RATE = 0.0003

        self.MAX_TRAINING_STEPS = self.NUM_ITEM * 2_000
        self.BUFFER_CAPACITY = self.NUM_ITEM * 2_000

        self.EPISODE_REWARD_AVG_SOLVED = 100
        self.EPISODE_REWARD_STD_SOLVED = 20

        self.CONSOLE_LOG_INTERVAL_TRAINING_STEPS = 50
        self.MODEL_TYPE = ModelType.TINY_1D_CONVOLUTIONAL
        self.TARGET_SYNC_INTERVAL_TRAINING_STEPS = 100


class ConfigNewKnapsack0StaticTestLinear(ConfigNewKnapsack):
    def __init__(self):
        ConfigNewKnapsack.__init__(self)

        self.NUM_ITEM = 50
        self.STATIC_INITIAL_STATE_50 = True
        self.LIMIT_WEIGHT_KNAPSACK = 200
        self.MODEL_TYPE = ModelType.MEDIUM_LINEAR

