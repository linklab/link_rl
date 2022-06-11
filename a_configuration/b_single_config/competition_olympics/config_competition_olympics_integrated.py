from a_configuration.a_base_config.a_environments.competition_olympics.config_competition_olympics_integrated import \
    ConfigCompetitionOlympicsIntegrated
from a_configuration.a_base_config.b_agents.config_agents_off_policy import ConfigSac
from a_configuration.a_base_config.b_agents.config_agents_on_policy import ConfigPpo
from a_configuration.a_base_config.config_single_base import ConfigBase
from g_utils.types import ModelType


class ConfigCompetitionOlympicsIntegratedPpo(ConfigBase, ConfigCompetitionOlympicsIntegrated, ConfigPpo):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigCompetitionOlympicsIntegrated.__init__(self)
        ConfigPpo.__init__(self)

        self.MAX_TRAINING_STEPS = 1_000_000
        self.MODEL_TYPE = ModelType.MEDIUM_2D_CONVOLUTIONAL


class ConfigCompetitionOlympicsIntegratedSac(ConfigBase, ConfigCompetitionOlympicsIntegrated, ConfigSac):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigCompetitionOlympicsIntegrated.__init__(self)
        ConfigSac.__init__(self)

        self.MAX_TRAINING_STEPS = 1_000_000
        self.MODEL_TYPE = ModelType.MEDIUM_2D_CONVOLUTIONAL
