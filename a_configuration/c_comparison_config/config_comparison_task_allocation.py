from a_configuration.a_base_config.config_comparison_base import ConfigComparisonBase
from a_configuration.b_single_config.config_task_allocation import ConfigTaskAllocation0Dqn, \
    ConfigTaskAllocation0DoubleDqn, ConfigTaskAllocation0DuelingDqn, ConfigTaskAllocation0DoubleDuelingDqn


class ConfigComparisonTaskAllocationDqnTypes(ConfigComparisonBase):
    def __init__(self):
        ConfigComparisonBase.__init__(self)

        self.ENV_NAME = "Task_Allocation_v0"

        self.AGENT_PARAMETERS = [
            ConfigTaskAllocation0Dqn(),
            ConfigTaskAllocation0DoubleDqn(),
            ConfigTaskAllocation0DuelingDqn(),
            ConfigTaskAllocation0DoubleDuelingDqn()
        ]

        self.AGENT_LABELS = [
            "DQN",
            "Double DQN",
            "Dueling DQN",
            "Double Dueling DQN",
        ]
        self.MAX_TRAINING_STEPS = 200_000
        self.N_RUNS = 5

