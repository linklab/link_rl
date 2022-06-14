from link_rl.a_configuration.a_base_config.config_comparison_base import ConfigComparisonBase
from link_rl.a_configuration.b_single_config.combinatorial_optimization.config_knapsack import \
    ConfigKnapsack0StaticTestLinearDqn, \
    ConfigKnapsack0StaticTestLinearA2c, ConfigKnapsack0StaticTestLinearPpo, \
    ConfigKnapsack0StaticTestLinearDoubleDuelingDqn, ConfigKnapsack0StaticTestLinearDoubleDqn, \
    ConfigKnapsack0RandomTestLinearDoubleDqn
from link_rl.g_utils.types import ModelType


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
            ConfigKnapsack0StaticTestLinearDoubleDqn(),
            ConfigKnapsack0StaticTestLinearDoubleDqn(),
        ]
        self.AGENT_PARAMETERS[1].MODEL_TYPE = ModelType.SMALL_RECURRENT

        self.AGENT_LABELS = [
            "Double DQN",
            "Recurrent Double DQN",
        ]

        self.MAX_TRAINING_STEPS = 200_000
        self.N_RUNS = 5


class ConfigComparisonKnapsack0StaticTestLinearDoubleDqn(ConfigComparisonBase):
    def __init__(self):
        ConfigComparisonBase.__init__(self)

        self.ENV_NAME = "Knapsack_Problem_v0"

        self.AGENT_PARAMETERS = [
            ConfigKnapsack0StaticTestLinearDoubleDqn(),
            ConfigKnapsack0StaticTestLinearDoubleDqn(),
        ]

        self.AGENT_LABELS = [
            "Static - Double DQN - Many Actions",
            "Static - Double DQN - Two Actions",
        ]
        self.AGENT_PARAMETERS[0].STRATEGY = 1
        self.AGENT_PARAMETERS[1].STRATEGY = 2

        self.MAX_TRAINING_STEPS = 500_000
        self.N_RUNS = 5


class ConfigComparisonKnapsack0RandomTestLinearDoubleDqn(ConfigComparisonBase):
    def __init__(self):
        ConfigComparisonBase.__init__(self)

        self.ENV_NAME = "Knapsack_Problem_v0"

        self.AGENT_PARAMETERS = [
            ConfigKnapsack0RandomTestLinearDoubleDqn(),
            ConfigKnapsack0RandomTestLinearDoubleDqn(),
        ]

        self.AGENT_LABELS = [
            "Random - Double DQN - Many Actions",
            "Random - Double DQN - Two Actions",
        ]
        self.AGENT_PARAMETERS[0].STRATEGY = 1
        self.AGENT_PARAMETERS[1].STRATEGY = 2

        self.MAX_TRAINING_STEPS = 1_000_000
        self.N_RUNS = 5


class ConfigComparisonKnapsack0RandomTestLinearDoubleDqnHer(ConfigComparisonBase):
    def __init__(self):
        ConfigComparisonBase.__init__(self)

        self.ENV_NAME = "Knapsack_Problem_v0"

        self.AGENT_PARAMETERS = [
            ConfigKnapsack0RandomTestLinearDoubleDqn(),
            ConfigKnapsack0RandomTestLinearDoubleDqn(),
        ]

        self.AGENT_LABELS = [
            "Random - Double DQN",
            "Random - Double DQN - HER",
        ]
        self.AGENT_PARAMETERS[1].USE_HER = True

        self.MAX_TRAINING_STEPS = 1_000_000
        self.N_RUNS = 5