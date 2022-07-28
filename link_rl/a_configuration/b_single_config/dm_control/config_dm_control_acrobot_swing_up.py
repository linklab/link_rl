from link_rl.a_configuration.a_base_config.a_environments.dm_control.config_dm_control_acrobot import \
    ConfigDmControlAcrobotSwingUp
from link_rl.a_configuration.a_base_config.b_agents.config_agents_off_policy import ConfigDdpg, ConfigTd3, ConfigSac, \
    ConfigTdmpc
from link_rl.a_configuration.a_base_config.b_agents.config_agents_on_policy import ConfigA2c, ConfigPpo, ConfigA3c
from link_rl.a_configuration.a_base_config.config_single_base import ConfigBase
from link_rl.d_models.d_basic_actor_critic_model import BASIC_ACTOR_CRITIC_MODEL
from link_rl.d_models.e_ddpg_model import DDPG_MODEL
from link_rl.d_models.f_td3_model import TD3_MODEL
from link_rl.d_models.g_sac_model import SAC_MODEL
from link_rl.d_models.h_tdmpc_model import TDMPC_MODEL


class ConfigDmControlAcrobotSwingUpA2c(ConfigBase, ConfigDmControlAcrobotSwingUp, ConfigA2c):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigDmControlAcrobotSwingUp.__init__(self)
        ConfigA2c.__init__(self)

        self.MAX_TRAINING_STEPS = 1_000_000
        self.MODEL_TYPE = BASIC_ACTOR_CRITIC_MODEL.ContinuousBasicActorCriticSharedModel.value


class ConfigDmControlAcrobotSwingUpA3c(ConfigBase, ConfigDmControlAcrobotSwingUp, ConfigA3c):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigDmControlAcrobotSwingUp.__init__(self)
        ConfigA3c.__init__(self)

        self.MAX_TRAINING_STEPS = 1_000_000
        self.MODEL_TYPE = BASIC_ACTOR_CRITIC_MODEL.ContinuousBasicActorCriticSharedModel.value


class ConfigDmControlAcrobotSwingUpPpo(ConfigBase, ConfigDmControlAcrobotSwingUp, ConfigPpo):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigDmControlAcrobotSwingUp.__init__(self)
        ConfigPpo.__init__(self)

        self.MAX_TRAINING_STEPS = 1_000_000
        self.MODEL_TYPE = BASIC_ACTOR_CRITIC_MODEL.ContinuousBasicActorCriticSharedModel.value


class ConfigDmControlAcrobotSwingUpDdpg(ConfigBase, ConfigDmControlAcrobotSwingUp, ConfigDdpg):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigDmControlAcrobotSwingUp.__init__(self)
        ConfigDdpg.__init__(self)

        self.MAX_TRAINING_STEPS = 1_000_000
        self.MODEL_TYPE = DDPG_MODEL.ContinuousDdpgModel.value


class ConfigDmControlAcrobotSwingUpTd3(ConfigBase, ConfigDmControlAcrobotSwingUp, ConfigTd3):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigDmControlAcrobotSwingUp.__init__(self)
        ConfigTd3.__init__(self)

        self.MAX_TRAINING_STEPS = 1_000_000
        self.MODEL_TYPE = TD3_MODEL.ContinuousTd3Model.value


class ConfigDmControlAcrobotSwingUpSac(ConfigBase, ConfigDmControlAcrobotSwingUp, ConfigSac):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigDmControlAcrobotSwingUp.__init__(self)
        ConfigSac.__init__(self)

        self.MAX_TRAINING_STEPS = 1_000_000
        self.MODEL_TYPE = SAC_MODEL.ContinuousSacModel.value


class ConfigDmControlAcrobotSwingUpTdmpc(ConfigBase, ConfigDmControlAcrobotSwingUp, ConfigTdmpc):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigDmControlAcrobotSwingUp.__init__(self)
        ConfigTdmpc.__init__(self)

        self.MAX_TRAINING_STEPS = 1_000_000
        self.MODEL_TYPE = TDMPC_MODEL.TdmpcModel.value
