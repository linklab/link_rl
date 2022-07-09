from link_rl.b_environments.task_allocation.task_allocation_env import TaskAllocationEnvironmentStat
from link_rl.c_encoders.a_encoder import ENCODER


class ConfigTaskAllocation:
    def __init__(self):
        # cloud network config
        self.NUM_CLOUD_SERVER = 15
        self.CLOUD_CPU_CAPACITY_MIN = 50
        self.CLOUD_CPU_CAPACITY_MAX = 100
        self.CLOUD_BANDWIDTH_CAPACITY = 100 * self.NUM_CLOUD_SERVER

        # edge network config
        self.NUM_EDGE_SERVER = 15
        self.EDGE_CPU_CAPACITY_MIN = 50
        self.EDGE_CPU_CAPACITY_MAX = 100
        self.EDGE_BANDWIDTH_CAPACITY = 100 * self.NUM_EDGE_SERVER

        # task config
        self.NUM_TASK = 200
        self.TASK_DATA_SIZE_MIN = 25
        self.TASK_DATA_SIZE_MAX = 50
        self.TASK_CPU_REQUEST_MIN = 10
        self.TASK_CPU_REQUEST_MAX = 20
        self.TASK_LATENCY_REQUEST_MIN = 40
        self.TASK_LATENCY_REQUEST_MAX = 70

        self.FIX_ENV_PARAM = 0

        #training setting
        self.MAX_TRAINING_STEPS = self.NUM_TASK * 2_000
        self.BUFFER_CAPACITY = self.NUM_TASK * 1_000
        self.EPISODE_REWARD_MIN_SOLVED = 1000000

        self.CONSOLE_LOG_INTERVAL_TRAINING_STEPS = 50

        self.ENCODER_TYPE = ENCODER.IdentityEncoder.value
        self.MODEL_TYPE = None
        self.TARGET_SYNC_INTERVAL_TRAINING_STEPS = 100

        self.CUSTOM_ENV_STAT = TaskAllocationEnvironmentStat()


class ConfigTaskAllocationInitParam(ConfigTaskAllocation):
    def __init__(self):
        ConfigTaskAllocation.__init__(self)

        self.ENV_NAME = "Task_Allocation_v1"
        self.LEARNING_RATE = 0.0003

        self.MODEL_TYPE = None
