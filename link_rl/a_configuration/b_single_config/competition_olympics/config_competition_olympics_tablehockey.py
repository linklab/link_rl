from link_rl.a_configuration.a_base_config.a_environments.competition_olympics.config_competition_olympics_integrated import \
    ConfigCompetitionOlympicsIntegrated
from link_rl.a_configuration.a_base_config.a_environments.competition_olympics.config_competition_olympics_tablehockey import \
    ConfigCompetitionOlympicsTableHockey
from link_rl.a_configuration.a_base_config.b_agents.config_agents_off_policy import ConfigSac, ConfigTd3
from link_rl.a_configuration.a_base_config.b_agents.config_agents_on_policy import ConfigPpo, ConfigA3c, ConfigAsynchronousPpo
from link_rl.a_configuration.a_base_config.config_single_base import ConfigBase
from link_rl.g_utils.types import ModelType


class ConfigCompetitionOlympicsTableHockeyA3c(ConfigBase, ConfigCompetitionOlympicsTableHockey, ConfigA3c):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigCompetitionOlympicsTableHockey.__init__(self)
        ConfigA3c.__init__(self)

        self.MAX_TRAINING_STEPS = 5_000_000
        self.MODEL_TYPE = ModelType.MEDIUM_2D_CONVOLUTIONAL
        self.TEST_INTERVAL_TRAINING_STEPS = 2_000
        self.BUFFER_CAPACITY = 1_000_000


class ConfigCompetitionOlympicsTableHockeyPpo(ConfigBase, ConfigCompetitionOlympicsTableHockey, ConfigPpo):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigCompetitionOlympicsTableHockey.__init__(self)
        ConfigPpo.__init__(self)

        self.MAX_TRAINING_STEPS = 5_000_000
        self.MODEL_TYPE = ModelType.MEDIUM_2D_CONVOLUTIONAL
        self.TEST_INTERVAL_TRAINING_STEPS = 2_000
        self.BUFFER_CAPACITY = 1_000_000


class ConfigCompetitionOlympicsTableHockeyAsynchronousPpo(ConfigBase, ConfigCompetitionOlympicsTableHockey, ConfigAsynchronousPpo):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigCompetitionOlympicsTableHockey.__init__(self)
        ConfigAsynchronousPpo.__init__(self)

        self.MAX_TRAINING_STEPS = 5_000_000
        self.MODEL_TYPE = ModelType.MEDIUM_2D_CONVOLUTIONAL
        self.TEST_INTERVAL_TRAINING_STEPS = 2_000
        self.BUFFER_CAPACITY = 1_000_000


class ConfigCompetitionOlympicsTableHockeyTd3(ConfigBase, ConfigCompetitionOlympicsTableHockey, ConfigTd3):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigCompetitionOlympicsTableHockey.__init__(self)
        ConfigTd3.__init__(self)

        self.MAX_TRAINING_STEPS = 5_000_000
        self.MODEL_TYPE = ModelType.MEDIUM_2D_CONVOLUTIONAL
        self.TEST_INTERVAL_TRAINING_STEPS = 2_000
        self.BUFFER_CAPACITY = 1_000_000


class ConfigCompetitionOlympicsTableHockeySac(ConfigBase, ConfigCompetitionOlympicsTableHockey, ConfigSac):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigCompetitionOlympicsTableHockey.__init__(self)
        ConfigSac.__init__(self)

        self.MAX_TRAINING_STEPS = 5_000_000
        self.MODEL_TYPE = ModelType.MEDIUM_2D_CONVOLUTIONAL
        self.TRAIN_INTERVAL_GLOBAL_TIME_STEPS = 10
        self.TEST_INTERVAL_TRAINING_STEPS = 2_000
        self.BUFFER_CAPACITY = 1_000_000
        self.ALPHA_LEARNING_RATE = 0.00005
