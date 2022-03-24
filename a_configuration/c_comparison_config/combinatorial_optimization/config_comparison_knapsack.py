from a_configuration.a_base_config.config_comparison_base import ConfigComparisonBase
from a_configuration.b_single_config.combinatorial_optimization.config_knapsack import ConfigKnapsack0Dqn, ConfigKnapsack0A2c, ConfigKnapsack0Ppo


class ConfigComparisonTaskAllocationDqnA2cPpo(ConfigComparisonBase):
    def __init__(self):
        ConfigComparisonBase.__init__(self)

        self.ENV_NAME = "Knapsack_Problem_v0"

        self.AGENT_PARAMETERS = [
            ConfigKnapsack0Dqn(),
            ConfigKnapsack0A2c(),
            ConfigKnapsack0Ppo(),
        ]

        self.AGENT_LABELS = [
            "DQN",
            "A2C",
            "PPO",
        ]

        self.MAX_TRAINING_STEPS = 200_000
        self.N_RUNS = 5

