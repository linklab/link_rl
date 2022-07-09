from link_rl.a_configuration.a_base_config.a_environments.dm_control.config_dm_control_humanoid import \
    ConfigDmControlHumanoidWalk
from link_rl.a_configuration.a_base_config.b_agents.config_agents_off_policy import ConfigSac, ConfigTdmpc
from link_rl.a_configuration.a_base_config.config_single_base import ConfigBase
from link_rl.d_models.g_sac_model import SAC_MODEL
from link_rl.d_models.h_tdmpc_model import TDMPC_MODEL


class ConfigDmControlHumanoidWalkTdmpc(ConfigBase, ConfigDmControlHumanoidWalk, ConfigTdmpc):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigDmControlHumanoidWalk.__init__(self)
        ConfigTdmpc.__init__(self)

        self.MAX_TRAINING_STEPS = 1_000_000
        self.ACTION_REPEAT = 2
        self.ITERATION = 12
        self.MODEL_TYPE = TDMPC_MODEL.TdmpcModel.value


class ConfigDmControlHumanoidWalkSac(ConfigBase, ConfigDmControlHumanoidWalk, ConfigSac):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigDmControlHumanoidWalk.__init__(self)
        ConfigSac.__init__(self)

        self.MAX_TRAINING_STEPS = 1_000_000
        self.ACTION_REPEAT = 2
        self.ITERATION = 12
        self.MODEL_TYPE = SAC_MODEL.ContinuousSacModel.value
