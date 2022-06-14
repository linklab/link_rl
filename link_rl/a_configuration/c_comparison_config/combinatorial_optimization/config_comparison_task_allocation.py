from link_rl.a_configuration.a_base_config.config_comparison_base import ConfigComparisonBase
from link_rl.a_configuration.b_single_config.combinatorial_optimization.config_basic_task_allocation import ConfigBasicTaskAllocation0Dqn, \
    ConfigBasicTaskAllocation0DoubleDqn, ConfigBasicTaskAllocation0DuelingDqn, ConfigBasicTaskAllocation0DoubleDuelingDqn


class ConfigComparisonTaskAllocationDqnTypes(ConfigComparisonBase):
    def __init__(self):
        ConfigComparisonBase.__init__(self)

        self.ENV_NAME = "Task_Allocation_v0"

        self.AGENT_PARAMETERS = [
            ConfigBasicTaskAllocation0Dqn(),
            ConfigBasicTaskAllocation0DoubleDqn(),
            ConfigBasicTaskAllocation0DuelingDqn(),
            ConfigBasicTaskAllocation0DoubleDuelingDqn()
        ]

        self.AGENT_LABELS = [
            "DQN",
            "Double DQN",
            "Dueling DQN",
            "Double Dueling DQN",
        ]
        self.MAX_TRAINING_STEPS = 200_000
        self.N_RUNS = 5

