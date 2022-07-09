from link_rl.a_configuration.a_base_config.a_environments.dm_control.config_dm_control_cheetah import ConfigDmControlCheetahRun
from link_rl.a_configuration.a_base_config.b_agents.config_agents_off_policy import ConfigDdpg, ConfigTd3, ConfigSac, \
    ConfigTdmpc
from link_rl.a_configuration.a_base_config.b_agents.config_agents_on_policy import ConfigA2c, ConfigPpo, ConfigA3c
from link_rl.a_configuration.a_base_config.config_single_base import ConfigBase
from link_rl.d_models.d_basic_actor_critic_model import BASIC_ACTOR_CRITIC_MODEL
from link_rl.d_models.e_ddpg_model import DDPG_MODEL
from link_rl.d_models.f_td3_model import TD3_MODEL
from link_rl.d_models.g_sac_model import SAC_MODEL
from link_rl.d_models.h_tdmpc_model import TDMPC_MODEL


class ConfigDmControlCheetahRunA2c(ConfigBase, ConfigDmControlCheetahRun, ConfigA2c):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigDmControlCheetahRun.__init__(self)
        ConfigA2c.__init__(self)

        self.MAX_TRAINING_STEPS = 1_000_000
        self.FROM_PIXELS = True
        self.MODEL_TYPE = BASIC_ACTOR_CRITIC_MODEL.ContinuousBasicActorCriticSharedModel.value


class ConfigDmControlCheetahRunA3c(ConfigBase, ConfigDmControlCheetahRun, ConfigA3c):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigDmControlCheetahRun.__init__(self)
        ConfigA3c.__init__(self)

        self.MAX_TRAINING_STEPS = 1_000_000
        self.FROM_PIXELS = True
        self.MODEL_TYPE = BASIC_ACTOR_CRITIC_MODEL.ContinuousBasicActorCriticSharedModel.value


class ConfigDmControlCheetahRunPpo(ConfigBase, ConfigDmControlCheetahRun, ConfigPpo):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigDmControlCheetahRun.__init__(self)
        ConfigPpo.__init__(self)

        self.MAX_TRAINING_STEPS = 1_000_000
        self.FROM_PIXELS = True
        self.MODEL_TYPE = BASIC_ACTOR_CRITIC_MODEL.ContinuousBasicActorCriticSharedModel.value


class ConfigDmControlCheetahRunDdpg(ConfigBase, ConfigDmControlCheetahRun, ConfigDdpg):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigDmControlCheetahRun.__init__(self)
        ConfigDdpg.__init__(self)

        self.MAX_TRAINING_STEPS = 1_000_000
        self.FROM_PIXELS = True
        self.MODEL_TYPE = DDPG_MODEL.ContinuousDdpgModel.value


class ConfigDmControlCheetahRunTd3(ConfigBase, ConfigDmControlCheetahRun, ConfigTd3):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigDmControlCheetahRun.__init__(self)
        ConfigTd3.__init__(self)

        self.MAX_TRAINING_STEPS = 1_000_000
        self.FROM_PIXELS = True
        self.MODEL_TYPE = TD3_MODEL.ContinuousTd3Model.value


class ConfigDmControlCheetahRunSac(ConfigBase, ConfigDmControlCheetahRun, ConfigSac):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigDmControlCheetahRun.__init__(self)
        ConfigSac.__init__(self)

        self.MAX_TRAINING_STEPS = 1_000_000
        self.FROM_PIXELS = True
        self.MODEL_TYPE = SAC_MODEL.ContinuousSacModel.value


class ConfigDmControlCheetahRunTdmpc(ConfigBase, ConfigDmControlCheetahRun, ConfigTdmpc):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigDmControlCheetahRun.__init__(self)
        ConfigTdmpc.__init__(self)

        self.MAX_TRAINING_STEPS = 1_000_000
        self.FROM_PIXELS = True
        self.MODEL_TYPE = TDMPC_MODEL.TdmpcModel.value
