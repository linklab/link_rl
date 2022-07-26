import os

from link_rl.b_environments.combinatorial_optimization.knapsack.knapsack import KnapsackEnvStat
from link_rl.c_encoders.a_encoder import ENCODER


class ConfigKnapsack:
    def __init__(self):
        self.ENV_NAME = "Knapsack_Problem_v0"

        self.PROJECT_HOME = os.path.abspath(
            os.path.join(
                os.path.dirname(os.path.realpath(__file__)), os.pardir, os.pardir, os.pardir, os.pardir, os.pardir
            )
        )

        self.COMBINATORIAL_OPTIMIZATION_ENV_DIR = os.path.join(
            self.PROJECT_HOME,
            "b_environments",
            "combinatorial_optimization"
        )
        if not os.path.exists(self.COMBINATORIAL_OPTIMIZATION_ENV_DIR):
            os.mkdir(self.COMBINATORIAL_OPTIMIZATION_ENV_DIR)

        self.NUM_ITEM = 20
        self.LIMIT_WEIGHT_KNAPSACK = 200

        self.MIN_WEIGHT_ITEM = 10
        self.MAX_WEIGHT_ITEM = 15

        self.MIN_VALUE_ITEM = 10
        self.MAX_VALUE_ITEM = 15

        self.INITIAL_ITEM_DISTRIBUTION_FIXED = True

        self.SORTING_TYPE = 1

        self.INITIAL_STATE_FILE_PATH = False
        self.UPLOAD_PATH = False
        self.OPTIMAL_PATH = False
        self.INSTANCE_INDEX = False

        self.STATIC_INITIAL_STATE_50 = False

        self.LEARNING_RATE = 0.0003

        self.MAX_TRAINING_STEPS = self.NUM_ITEM * 2_000
        self.BUFFER_CAPACITY = self.NUM_ITEM * 2_000

        self.EPISODE_REWARD_MEAN_SOLVED = 100

        self.CONSOLE_LOG_INTERVAL_TRAINING_STEPS = 50

        self.ENCODER_TYPE = ENCODER.IdentityEncoder.value
        self.MODEL_TYPE = None

        self.TARGET_SYNC_INTERVAL_TRAINING_STEPS = 100

        self.PRINT_DETAILS_AT_EPISODE_END = False

        self.CUSTOM_ENV_STAT = KnapsackEnvStat()


class ConfigKnapsack0(ConfigKnapsack):
    def __init__(self):
        ConfigKnapsack.__init__(self)

        self.STRATEGY = 2  # ACTION SPACE: 2


class ConfigKnapsack0RandomTest(ConfigKnapsack0):
    def __init__(self):
        ConfigKnapsack0.__init__(self)


class ConfigKnapsack0RandomTestLinear(ConfigKnapsack0):
    def __init__(self):
        ConfigKnapsack0.__init__(self)


class ConfigKnapsack0LoadTest(ConfigKnapsack0):
    def __init__(self):
        ConfigKnapsack0.__init__(self)

        self.NUM_ITEM = 50
        self.INITIAL_STATE_FILE_PATH = 'knapsack_instances/RI/instances/n_50_r_100'
        self.UPLOAD_PATH = 'knapsack_instances/RI/link_solution/n_50_r_100'
        self.OPTIMAL_PATH = 'knapsack_instances/RI/optimal_solution/n_50_r_100'
        self.INSTANCE_INDEX = 0


class ConfigKnapsack0LoadTestLinear(ConfigKnapsack0):
    def __init__(self):
        ConfigKnapsack0.__init__(self)

        self.NUM_ITEM = 50
        self.INITIAL_STATE_FILE_PATH = 'knapsack_instances/RI/instances/n_50_r_100'
        self.UPLOAD_PATH = 'knapsack_instances/RI/link_solution/n_50_r_100'
        self.OPTIMAL_PATH = 'knapsack_instances/RI/optimal_solution/n_50_r_100'
        self.INSTANCE_INDEX = 0


class ConfigKnapsack0StaticTest(ConfigKnapsack0):
    def __init__(self):
        ConfigKnapsack0.__init__(self)

        self.NUM_ITEM = 50
        self.STATIC_INITIAL_STATE_50 = True


class ConfigKnapsack0StaticTestLinear(ConfigKnapsack0):
    def __init__(self):
        ConfigKnapsack0.__init__(self)

        self.NUM_ITEM = 50
        self.STATIC_INITIAL_STATE_50 = True


class ConfigKnapsack0StaticTestLinearRecurrent(ConfigKnapsack0):
    def __init__(self):
        ConfigKnapsack0.__init__(self)

        self.NUM_ITEM = 50
        self.STATIC_INITIAL_STATE_50 = True
