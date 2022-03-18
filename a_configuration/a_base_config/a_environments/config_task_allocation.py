from g_utils.types import LossFunctionType


class ConfigTakAllocation:
    pass


class ConfigTakAllocation0(ConfigTakAllocation):
    def __init__(self):
        self.ENV_NAME = "TaskAllocation_v0"
        self.EPISODE_REWARD_AVG_SOLVED = 100
        self.EPISODE_REWARD_STD_SOLVED = 20
        self.NUM_TASK = 10
        self.NUM_RESOURCES = 1
        self.INITIAL_RESOURCES_CAPACITY = [100]
        self.LOW_DEMAND_RESOURCE_AT_TASK = [10]
        self.HIGH_DEMAND_RESOURCE_AT_TASK = [12]
        self.INITIAL_TASK_DISTRIBUTION_FIXED = True
        