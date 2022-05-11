from g_utils.types import ModelType

class ConfigTaskAllocation:
    def __init__(self):
        # cloud network config
        self.NUM_CLOUD_SERVER = 3
        self.CLOUD_CPU_CAPACITY_MIN = 50
        self.CLOUD_CPU_CAPACITY_MAX = 100
        self.CLOUD_BANDWIDTH_CAPACITY = 10000

        # edge network config
        self.NUM_EDGE_SERVER = 3
        self.EDGE_CPU_CAPACITY_MIN = 50
        self.EDGE_CPU_CAPACITY_MAX = 100
        self.EDGE_BANDWIDTH_CAPACITY = 10000

        # task config
        self.NUM_TASK = 3
        self.TASK_DATA_SIZE_MIN = 25
        self.TASK_DATA_SIZE_MAX = 50
        self.TASK_CPU_REQUEST_MIN = 10
        self.TASK_CPU_REQUEST_MAX = 20
        self.TASK_LATENCY_REQUEST_MIN = 1
        self.TASK_LATENCY_REQUEST_MAX = 100



