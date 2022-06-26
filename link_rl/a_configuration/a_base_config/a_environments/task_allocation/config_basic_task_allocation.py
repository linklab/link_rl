from link_rl.b_environments.task_allocation.basic_task_allocation import EnvironmentBasicTaskScheduling0Stat
from link_rl.g_utils.types import ModelType


class ConfigBasicTaskAllocation:
    def __init__(self):
        self.EPISODE_REWARD_MIN_SOLVED = 100

        self.CONSOLE_LOG_INTERVAL_TRAINING_STEPS = 50
        self.MODEL_TYPE = ModelType.SMALL_LINEAR
        self.TARGET_SYNC_INTERVAL_TRAINING_STEPS = 100

        self.NUM_TASK = 20

        self.LOW_DEMAND_RESOURCE_AT_TASK = [10]
        self.HIGH_DEMAND_RESOURCE_AT_TASK = [15]
        self.INITIAL_TASK_DISTRIBUTION_FIXED = True
        self.MAX_TRAINING_STEPS = self.NUM_TASK * 2_000
        self.BUFFER_CAPACITY = self.NUM_TASK * 1_000

        self.CUSTOM_ENV_STAT = EnvironmentBasicTaskScheduling0Stat()


class ConfigBasicTaskAllocation0(ConfigBasicTaskAllocation):
    def __init__(self):
        ConfigBasicTaskAllocation.__init__(self)

        self.ENV_NAME = "Task_Allocation_v0"
        self.LEARNING_RATE = 0.0003

        self.NUM_RESOURCES = 1
        self.INITIAL_RESOURCES_CAPACITY = [200]

