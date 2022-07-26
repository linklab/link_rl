from link_rl.a_configuration.a_base_config.a_environments.somo_gym.config_somo_gym_in_hand_manipulation import \
    ConfigSomoGymInHandManipulation
from link_rl.a_configuration.a_base_config.b_agents.config_agents_off_policy import ConfigSac, ConfigTdmpc
from link_rl.a_configuration.a_base_config.config_single_base import ConfigBase
from link_rl.d_models.g_sac_model import SAC_MODEL
from link_rl.d_models.h_tdmpc_model import TDMPC_MODEL


class ConfigSomoGymInHandManipulationSac(ConfigBase, ConfigSomoGymInHandManipulation, ConfigSac):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigSomoGymInHandManipulation.__init__(self)
        ConfigSac.__init__(self)

        self.MAX_TRAINING_STEPS = 1_000_000
        self.BUFFER_CAPACITY = 1_000_000
        self.ALPHA_LEARNING_RATE = 0.000025
        self.MIN_ALPHA = 0.2
        self.MODEL_TYPE = SAC_MODEL.ContinuousSacModel.value


class ConfigSomoGymInHandManipulationTdmpc(ConfigBase, ConfigSomoGymInHandManipulation, ConfigTdmpc):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigSomoGymInHandManipulation.__init__(self)
        ConfigTdmpc.__init__(self)

        self.MAX_TRAINING_STEPS = 1_000_000
        self.MODEL_TYPE = TDMPC_MODEL.TdmpcModel.value
