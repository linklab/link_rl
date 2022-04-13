from a_configuration.a_base_config.config_comparison_base import ConfigComparisonBase
from a_configuration.b_single_config.open_ai_gym.atari.config_pong import ConfigPongDqn, ConfigPongDoubleDqn, \
    ConfigPongDuelingDqn, ConfigPongDoubleDuelingDqn


class ConfigComparisonPongDqn(ConfigComparisonBase):
    def __init__(self):
        ConfigComparisonBase.__init__(self)
        self.ENV_NAME = "ALE/Pong-v5"

        self.MAX_TRAINING_STEPS = 1_000_000

        self.AGENT_PARAMETERS = [
            ConfigPongDqn(),
            ConfigPongDqn(),
            ConfigPongDqn()
        ]

        self.AGENT_PARAMETERS[0].N_STEP = 1
        self.AGENT_PARAMETERS[1].N_STEP = 2
        self.AGENT_PARAMETERS[2].N_STEP = 3
        self.AGENT_LABELS = [
            "DQN (N_STEP=1)",
            "DQN (N_STEP=2)",
            "DQN (N_STEP=3)",
        ]
        self.MAX_TRAINING_STEPS = 1_000
        self.N_RUNS = 5


class ConfigComparisonPongDqnTypes(ConfigComparisonBase):
    def __init__(self):
        ConfigComparisonBase.__init__(self)

        self.ENV_NAME = "ALE/Pong-v5"

        self.MAX_TRAINING_STEPS = 1_000_000

        self.AGENT_PARAMETERS = [
            ConfigPongDqn(),
            ConfigPongDoubleDqn(),
            ConfigPongDuelingDqn(),
            ConfigPongDoubleDuelingDqn(),
            ConfigPongDoubleDuelingDqn()
        ]

        self.AGENT_PARAMETERS[4].USE_PER = True

        self.AGENT_LABELS = [
            "DQN",
            "Double DQN",
            "Dueling DQN",
            "Double Dueling DQN",
            "Double Dueling DQN + PER",
        ]

        self.N_RUNS = 3