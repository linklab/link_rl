from link_rl.a_configuration.a_base_config.a_environments.evolution_gym.config_evolution_gym_walker import \
    ConfigEvolutionGymWalker, ConfigEvolutionGymBridgeWalker, ConfigEvolutionGymCaveCrawler
from link_rl.a_configuration.a_base_config.b_agents.config_agents_off_policy import ConfigSac, ConfigTdmpc
from link_rl.a_configuration.a_base_config.config_single_base import ConfigBase
from link_rl.d_models.g_sac_model import SAC_MODEL
from link_rl.d_models.h_tdmpc_model import TDMPC_MODEL


class ConfigEvolutionGymWalkerSac(ConfigBase, ConfigEvolutionGymWalker, ConfigSac):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigEvolutionGymWalker.__init__(self)
        ConfigSac.__init__(self)

        self.MAX_TRAINING_STEPS = 1_000_000
        self.BUFFER_CAPACITY = 500_000
        self.ALPHA_LEARNING_RATE = 0.000025
        self.MIN_ALPHA = 0.2
        self.MODEL_TYPE = SAC_MODEL.ContinuousSacModel.value


class ConfigEvolutionGymBridgeWalkerSac(ConfigBase, ConfigEvolutionGymBridgeWalker, ConfigSac):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigEvolutionGymBridgeWalker.__init__(self)
        ConfigSac.__init__(self)

        self.MAX_TRAINING_STEPS = 1_000_000
        self.BUFFER_CAPACITY = 500_000
        self.ALPHA_LEARNING_RATE = 0.000025
        self.MIN_ALPHA = 0.2
        self.MODEL_TYPE = SAC_MODEL.ContinuousSacModel.value


class ConfigEvolutionGymCaveCrawlerSac(ConfigBase, ConfigEvolutionGymCaveCrawler, ConfigSac):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigEvolutionGymCaveCrawler.__init__(self)
        ConfigSac.__init__(self)

        self.MAX_TRAINING_STEPS = 1_000_000
        self.BUFFER_CAPACITY = 500_000
        self.ALPHA_LEARNING_RATE = 0.000025
        self.MIN_ALPHA = 0.2
        self.MODEL_TYPE = SAC_MODEL.ContinuousSacModel.value


class ConfigEvolutionGymWalkerTdmpc(ConfigBase, ConfigEvolutionGymWalker, ConfigTdmpc):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigEvolutionGymWalker.__init__(self)
        ConfigTdmpc.__init__(self)

        self.MAX_TRAINING_STEPS = 1_000_000
        self.MODEL_TYPE = TDMPC_MODEL.TdmpcModel.value


class ConfigEvolutionGymBridgeWalkerTdmpc(ConfigBase, ConfigEvolutionGymBridgeWalker, ConfigTdmpc):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigEvolutionGymBridgeWalker.__init__(self)
        ConfigTdmpc.__init__(self)

        self.MAX_TRAINING_STEPS = 1_000_000
        self.MODEL_TYPE = TDMPC_MODEL.TdmpcModel.value


class ConfigEvolutionGymCaveCrawlerTdmpc(ConfigBase, ConfigEvolutionGymCaveCrawler, ConfigTdmpc):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigEvolutionGymCaveCrawler.__init__(self)
        ConfigTdmpc.__init__(self)

        self.MAX_TRAINING_STEPS = 1_000_000
        self.MODEL_TYPE = TDMPC_MODEL.TdmpcModel.value


