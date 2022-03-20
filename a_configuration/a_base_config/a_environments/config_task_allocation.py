from g_utils.types import ModelType


class ConfigTakAllocation:
    pass


class ConfigTakAllocation0(ConfigTakAllocation):
    def __init__(self):
        self.ENV_NAME = "Task_Allocation_v0"
        self.EPISODE_REWARD_AVG_SOLVED = 100
        self.EPISODE_REWARD_STD_SOLVED = 20
        self.NUM_TASK = 20

        self.NUM_RESOURCES = 1
        self.INITIAL_RESOURCES_CAPACITY = [200]
        self.LOW_DEMAND_RESOURCE_AT_TASK = [10]
        self.HIGH_DEMAND_RESOURCE_AT_TASK = [15]
        self.INITIAL_TASK_DISTRIBUTION_FIXED = True

        self.LEARNING_RATE = 0.0003
        self.MAX_TRAINING_STEPS = self.NUM_TASK * 2_000
        self.BUFFER_CAPACITY = 35_000
        self.CONSOLE_LOG_INTERVAL_TRAINING_STEPS = 50
        self.MODEL_TYPE = ModelType.SMALL_LINEAR
        self.TARGET_SYNC_INTERVAL_TRAINING_STEPS = 100
