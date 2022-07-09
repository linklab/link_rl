from link_rl.a_configuration.a_base_config.a_environments.competition_olympics.config_competition_olympics_integrated \
    import ConfigCompetitionOlympicsIntegrated
from link_rl.a_configuration.a_base_config.b_agents.config_agents_off_policy import ConfigSac, ConfigTd3
from link_rl.a_configuration.a_base_config.b_agents.config_agents_on_policy import ConfigPpo, ConfigA3c, \
    ConfigAsynchronousPpo
from link_rl.a_configuration.a_base_config.config_single_base import ConfigBase
from link_rl.d_models.d_basic_actor_critic_model import BASIC_ACTOR_CRITIC_MODEL
from link_rl.d_models.f_td3_model import TD3_MODEL
from link_rl.d_models.g_sac_model import SAC_MODEL


class ConfigCompetitionOlympicsIntegratedA3c(ConfigBase, ConfigCompetitionOlympicsIntegrated, ConfigA3c):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigCompetitionOlympicsIntegrated.__init__(self)
        ConfigA3c.__init__(self)

        self.MAX_TRAINING_STEPS = 5_000_000
        self.TEST_INTERVAL_TRAINING_STEPS = 2_000
        self.MODEL_TYPE = BASIC_ACTOR_CRITIC_MODEL.ContinuousBasicActorCriticSharedModel.value


class ConfigCompetitionOlympicsIntegratedPpo(ConfigBase, ConfigCompetitionOlympicsIntegrated, ConfigPpo):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigCompetitionOlympicsIntegrated.__init__(self)
        ConfigPpo.__init__(self)

        self.MAX_TRAINING_STEPS = 5_000_000
        self.TEST_INTERVAL_TRAINING_STEPS = 2_000
        self.MODEL_TYPE = BASIC_ACTOR_CRITIC_MODEL.ContinuousBasicActorCriticSharedModel.value


class ConfigCompetitionOlympicsIntegratedAsynchronousPpo(ConfigBase, ConfigCompetitionOlympicsIntegrated, ConfigAsynchronousPpo):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigCompetitionOlympicsIntegrated.__init__(self)
        ConfigAsynchronousPpo.__init__(self)

        self.MAX_TRAINING_STEPS = 5_000_000
        self.TEST_INTERVAL_TRAINING_STEPS = 2_000
        self.MODEL_TYPE = BASIC_ACTOR_CRITIC_MODEL.ContinuousBasicActorCriticSharedModel.value


class ConfigCompetitionOlympicsIntegratedTd3(ConfigBase, ConfigCompetitionOlympicsIntegrated, ConfigTd3):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigCompetitionOlympicsIntegrated.__init__(self)
        ConfigTd3.__init__(self)

        self.MAX_TRAINING_STEPS = 5_000_000
        self.TEST_INTERVAL_TRAINING_STEPS = 2_000
        self.MODEL_TYPE = TD3_MODEL.ContinuousTd3Model.value


class ConfigCompetitionOlympicsIntegratedSac(ConfigBase, ConfigCompetitionOlympicsIntegrated, ConfigSac):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigCompetitionOlympicsIntegrated.__init__(self)
        ConfigSac.__init__(self)

        self.MAX_TRAINING_STEPS = 5_000_000
        self.TRAIN_INTERVAL_GLOBAL_TIME_STEPS = 10
        self.TEST_INTERVAL_TRAINING_STEPS = 2_000
        self.ALPHA_LEARNING_RATE = 0.000025
        self.OPPONENT_AGENT_RANDOM_ACTION_RATIO = 'linear(0.3, 0.01, 100000)'
        self.MIN_ALPHA = 0.2
        self.MODEL_TYPE = SAC_MODEL.ContinuousSacModel.value
