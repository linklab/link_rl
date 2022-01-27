from a_configuration.a_base_config.config_comparison_base import ConfigComparisonBase
from a_configuration.b_single_config.open_ai_gym.config_lunar_lander import ConfigLunarLanderDqn
from g_utils.types import ModelType


class ConfigComparisonLunarLanderDqnRecurrent(ConfigComparisonBase):
    def __init__(self):
        ConfigComparisonBase.__init__(self)

        self.ENV_NAME = "LunarLander-v2"

        self.AGENT_PARAMETERS = [
            ConfigLunarLanderDqn(),
            ConfigLunarLanderDqn()
        ]

        self.AGENT_PARAMETERS[0].MODEL_TYPE = ModelType.SMALL_LINEAR
        self.AGENT_PARAMETERS[1].MODEL_TYPE = ModelType.SMALL_RECURRENT

        self.AGENT_LABELS = [
            "DQN Small Linear",
            "DQN Small Recurrent",
        ]

        self.MAX_TRAINING_STEPS = 100_000
        self.N_RUNS = 5

