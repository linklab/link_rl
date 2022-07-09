from link_rl.a_configuration.a_base_config.a_environments.dm_control.config_dm_control_finger import \
    ConfigDmControlFingerSpin
from link_rl.a_configuration.a_base_config.b_agents.config_agents_off_policy import ConfigSac, \
    ConfigTdmpc
from link_rl.a_configuration.a_base_config.config_single_base import ConfigBase
from link_rl.c_models_v2.g_sac_model import SAC_MODEL
from link_rl.c_models_v2.h_tdmpc_model import TDMPC_MODEL


class ConfigDmControlFingerSpinTdmpc(ConfigBase, ConfigDmControlFingerSpin, ConfigTdmpc):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigDmControlFingerSpin.__init__(self)
        ConfigTdmpc.__init__(self)

        self.MAX_TRAINING_STEPS = 1_000_000
        self.MODEL_TYPE = TDMPC_MODEL.TdmpcModel.value


class ConfigDmControlFingerSpinSac(ConfigBase, ConfigDmControlFingerSpin, ConfigSac):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigDmControlFingerSpin.__init__(self)
        ConfigSac.__init__(self)

        self.MAX_TRAINING_STEPS = 1_000_000
        self.MODEL_TYPE = SAC_MODEL.ContinuousSacModel.value
