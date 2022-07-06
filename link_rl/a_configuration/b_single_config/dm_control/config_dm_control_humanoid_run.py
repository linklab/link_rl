from link_rl.a_configuration.a_base_config.a_environments.dm_control.config_dm_control_cheetah import ConfigDmControlCheetahRun
from link_rl.a_configuration.a_base_config.a_environments.dm_control.config_dm_control_finger import \
    ConfigDmControlFingerSpin
from link_rl.a_configuration.a_base_config.a_environments.dm_control.config_dm_control_humanoid import \
    ConfigDmControlHumanoidWalk, ConfigDmControlHumanoidRun
from link_rl.a_configuration.a_base_config.b_agents.config_agents_off_policy import ConfigDdpg, ConfigTd3, ConfigSac, \
    ConfigTdmpc
from link_rl.a_configuration.a_base_config.b_agents.config_agents_on_policy import ConfigA2c, ConfigPpo, ConfigA3c
from link_rl.a_configuration.a_base_config.config_single_base import ConfigBase
from link_rl.c_models_v2.g_sac_model import SAC_MODEL
from link_rl.c_models_v2.h_tdmpc_model import TDMPC_MODEL
from link_rl.g_utils.types import ModelType


class ConfigDmControlHumanoidRunTdmpc(ConfigBase, ConfigDmControlHumanoidRun, ConfigTdmpc):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigDmControlHumanoidRun.__init__(self)
        ConfigTdmpc.__init__(self)

        self.MAX_TRAINING_STEPS = 1_000_000
        self.ACTION_REPEAT = 2
        self.ITERATION = 12
        self.MODEL_TYPE = TDMPC_MODEL.TdmpcEncoderModel.value


class ConfigDmControlHumanoidRunSac(ConfigBase, ConfigDmControlHumanoidRun, ConfigSac):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigDmControlHumanoidRun.__init__(self)
        ConfigSac.__init__(self)

        self.MAX_TRAINING_STEPS = 1_000_000
        self.ACTION_REPEAT = 2
        self.ITERATION = 12
        self.MODEL_TYPE = SAC_MODEL.ContinuousSacModel.value
