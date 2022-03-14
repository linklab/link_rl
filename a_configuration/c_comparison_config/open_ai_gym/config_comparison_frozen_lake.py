from a_configuration.a_base_config.a_environments.open_ai_gym.config_gym_toy_text import ConfigFrozenLake
from a_configuration.a_base_config.config_comparison_base import ConfigComparisonBase
from a_configuration.b_single_config.open_ai_gym.config_cart_pole import ConfigCartPoleDqn, \
    ConfigCartPoleReinforce, \
    ConfigCartPoleA2c, ConfigCartPoleDoubleDqn, ConfigCartPoleDuelingDqn, ConfigCartPoleDoubleDuelingDqn, \
    ConfigCartPolePpo, ConfigCartPolePpoTrajectory
from a_configuration.b_single_config.open_ai_gym.config_frozen_lake import ConfigFrozenLakeDqn


class ConfigComparisonFrozenLakeDqnActionMasking(ConfigComparisonBase):
    def __init__(self):
        ConfigComparisonBase.__init__(self)

        self.ENV_NAME = "FrozenLake-v1"

        self.AGENT_PARAMETERS = [
            ConfigFrozenLakeDqn(),
            ConfigFrozenLakeDqn()
        ]

        self.AGENT_PARAMETERS[0].ACTION_MASKING = True
        self.AGENT_PARAMETERS[0].RANDOM_MAP = False
        self.AGENT_PARAMETERS[1].ACTION_MASKING = False
        self.AGENT_PARAMETERS[1].RANDOM_MAP = False

        self.AGENT_LABELS = [
            "with Action Masking",
            "without Action Masking"
        ]
        self.MAX_TRAINING_STEPS = 100_000
        self.N_RUNS = 5

        self.RANDOM_MAP = False
        self.ACTION_MASKING = True

