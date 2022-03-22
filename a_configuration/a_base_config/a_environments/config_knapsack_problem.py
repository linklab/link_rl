from g_utils.types import ModelType


class ConfigKnapsack:
    def __init__(self):
        self.EPISODE_REWARD_AVG_SOLVED = 100
        self.EPISODE_REWARD_STD_SOLVED = 20

        self.CONSOLE_LOG_INTERVAL_TRAINING_STEPS = 50
        self.MODEL_TYPE = ModelType.SMALL_LINEAR
        self.TARGET_SYNC_INTERVAL_TRAINING_STEPS = 100

        self.NUM_ITEM = 20

        self.MIN_WEIGHT_ITEM = 10
        self.MAX_WEIGHT_KNAPSACK_ITEM = 15

        self.MIN_VALUE_ITEM = 10
        self.MAX_VALUE_ITEM = 15

        self.INITIAL_ITEM_DISTRIBUTION_FIXED = True

        self.MAX_TRAINING_STEPS = self.NUM_ITEM * 2_000
        self.BUFFER_CAPACITY = self.NUM_ITEM * 1_000


class ConfigKnapsack0(ConfigKnapsack):
    def __init__(self):
        ConfigKnapsack.__init__(self)

        self.ENV_NAME = "Knapsack_Problem_v0"
        self.LEARNING_RATE = 0.0003

        self.LIMIT_WEIGHT_KNAPSACK = 200
