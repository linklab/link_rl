from a_configuration.a_base_config.config_comparison_base import ConfigComparisonBase
from a_configuration.b_single_config.open_ai_gym.config_cart_pole import ConfigCartPoleDqn, \
    ConfigCartPoleReinforce, \
    ConfigCartPoleA2c, ConfigCartPoleDoubleDqn, ConfigCartPoleDuelingDqn, ConfigCartPoleDoubleDuelingDqn, \
    ConfigCartPolePpo, ConfigCartPolePpoTrajectory


class ConfigComparisonCartPoleDqn(ConfigComparisonBase):
    def __init__(self):
        ConfigComparisonBase.__init__(self)

        self.ENV_NAME = "CartPole-v1"

        self.AGENT_PARAMETERS = [
            ConfigCartPoleDqn(),
            ConfigCartPoleDqn(),
            ConfigCartPoleDqn()
        ]

        self.AGENT_PARAMETERS[0].N_STEP = 1
        self.AGENT_PARAMETERS[1].N_STEP = 2
        self.AGENT_PARAMETERS[2].N_STEP = 4
        self.AGENT_LABELS = [
            "DQN (N_STEP=1)",
            "DQN (N_STEP=2)",
            "DQN (N_STEP=4)",
        ]
        self.MAX_TRAINING_STEPS = 50_000
        self.N_RUNS = 5


class ConfigComparisonCartPoleDqnTypes(ConfigComparisonBase):
    def __init__(self):
        ConfigComparisonBase.__init__(self)

        self.ENV_NAME = "CartPole-v1"

        self.AGENT_PARAMETERS = [
            ConfigCartPoleDqn(),
            ConfigCartPoleDoubleDqn(),
            ConfigCartPoleDuelingDqn(),
            ConfigCartPoleDoubleDuelingDqn()
        ]

        self.AGENT_LABELS = [
            "DQN",
            "Double DQN",
            "Dueling DQN",
            "Double Dueling DQN",
        ]
        self.MAX_TRAINING_STEPS = 50_000
        self.N_RUNS = 5

# OnPolicy
class ConfigComparisonCartPoleReinforce(ConfigComparisonBase):
    def __init__(self):
        ConfigComparisonBase.__init__(self)

        self.AGENT_PARAMETERS = [
            ConfigCartPoleReinforce(),
            ConfigCartPoleReinforce(),
            ConfigCartPoleReinforce()
        ]


class ConfigComparisonCartPoleA2c(ConfigComparisonBase):
    def __init__(self):
        ConfigComparisonBase.__init__(self)

        self.AGENT_PARAMETERS = [
            ConfigCartPoleA2c(),
            ConfigCartPoleA2c(),
            ConfigCartPoleA2c(),
        ]


class ConfigComparisonCartPolePpo(ConfigComparisonBase):
    def __init__(self):
        ConfigComparisonBase.__init__(self)

        self.AGENT_PARAMETERS = [
            ConfigCartPolePpo(),
            ConfigCartPolePpoTrajectory()
        ]

        self.AGENT_LABELS = [
            "PPO",
            "PPO Trajectory"
        ]
        self.MAX_TRAINING_STEPS = 10_000
        self.N_RUNS = 5
