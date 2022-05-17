from a_configuration.a_base_config.config_comparison_base import ConfigComparisonBase
from a_configuration.b_single_config.combinatorial_optimization.config_knapsack import \
    ConfigKnapsack0StaticTestLinearDqn, \
    ConfigKnapsack0StaticTestLinearA2c, ConfigKnapsack0StaticTestLinearPpo, \
    ConfigKnapsack0StaticTestLinearDoubleDuelingDqn, ConfigKnapsack0StaticTestLinearDoubleDqn
from g_utils.types import ModelType


class ConfigComparisonKnapsack0StaticTestLinearDqnA2cPpo(ConfigComparisonBase):
    def __init__(self):
        ConfigComparisonBase.__init__(self)

        self.ENV_NAME = "Knapsack_Problem_v0"

        self.AGENT_PARAMETERS = [
            ConfigKnapsack0StaticTestLinearDqn(),
            ConfigKnapsack0StaticTestLinearA2c(),
            ConfigKnapsack0StaticTestLinearPpo(),
        ]

        self.AGENT_LABELS = [
            "DQN",
            "A2C",
            "PPO",
        ]

        self.MAX_TRAINING_STEPS = 200_000
        self.N_RUNS = 5


class ConfigComparisonKnapsack0StaticTestLinearDqn(ConfigComparisonBase):
    def __init__(self):
        ConfigComparisonBase.__init__(self)

        self.ENV_NAME = "Knapsack_Problem_v0"

        self.AGENT_PARAMETERS = [
            ConfigKnapsack0StaticTestLinearDqn(),
            ConfigKnapsack0StaticTestLinearDoubleDqn(),
            ConfigKnapsack0StaticTestLinearDoubleDuelingDqn(),
        ]

        self.AGENT_LABELS = [
            "DQN",
            "Double DQN",
            "Double Dueling DQN",
        ]

        self.MAX_TRAINING_STEPS = 200_000
        self.N_RUNS = 5


class ConfigComparisonKnapsack0StaticTestLinearRecurrentDqn(ConfigComparisonBase):
    def __init__(self):
        ConfigComparisonBase.__init__(self)

        self.ENV_NAME = "Knapsack_Problem_v0"

        self.AGENT_PARAMETERS = [
            ConfigKnapsack0StaticTestLinearDoubleDuelingDqn(),
            ConfigKnapsack0StaticTestLinearDoubleDuelingDqn(),
        ]
        self.AGENT_PARAMETERS[1].MODEL_TYPE = ModelType.SMALL_RECURRENT

        self.AGENT_LABELS = [
            "Double Dueling DQN",
            "Recurrent Double Dueling DQN",
        ]

        self.MAX_TRAINING_STEPS = 200_000
        self.N_RUNS = 5