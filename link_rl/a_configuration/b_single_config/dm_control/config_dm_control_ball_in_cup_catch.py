from link_rl.a_configuration.a_base_config.a_environments.dm_control.config_dm_control_ball_in_cup import \
    ConfigDmControlBallInCupCatch
from link_rl.a_configuration.a_base_config.b_agents.config_agents_off_policy import ConfigDdpg, ConfigTd3, ConfigSac, \
    ConfigTdmpc
from link_rl.a_configuration.a_base_config.b_agents.config_agents_on_policy import ConfigA2c, ConfigPpo, ConfigA3c
from link_rl.a_configuration.a_base_config.config_single_base import ConfigBase
from link_rl.c_models_v2.d_basic_actor_critic_model import BASIC_ACTOR_CRITIC_MODEL
from link_rl.c_models_v2.e_ddpg_model import DDPG_MODEL
from link_rl.c_models_v2.f_td3_model import TD3_MODEL
from link_rl.c_models_v2.g_sac_model import SAC_MODEL
from link_rl.c_models_v2.h_tdmpc_model import TDMPC_MODEL


class ConfigDmControlBallInCupCatchA2c(ConfigBase, ConfigDmControlBallInCupCatch, ConfigA2c):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigDmControlBallInCupCatch.__init__(self)
        ConfigA2c.__init__(self)

        self.MAX_TRAINING_STEPS = 1_000_000
        self.MODEL_TYPE = BASIC_ACTOR_CRITIC_MODEL.ContinuousBasicActorCriticEncoderSharedModel.value


class ConfigDmControlBallInCupCatchA3c(ConfigBase, ConfigDmControlBallInCupCatch, ConfigA3c):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigDmControlBallInCupCatch.__init__(self)
        ConfigA3c.__init__(self)

        self.MAX_TRAINING_STEPS = 1_000_000
        self.FROM_PIXELS = True
        self.MODEL_TYPE = BASIC_ACTOR_CRITIC_MODEL.ContinuousBasicActorCriticEncoderSharedModel.value


class ConfigDmControlBallInCupCatchPpo(ConfigBase, ConfigDmControlBallInCupCatch, ConfigPpo):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigDmControlBallInCupCatch.__init__(self)
        ConfigPpo.__init__(self)

        self.MAX_TRAINING_STEPS = 1_000_000
        self.FROM_PIXELS = True
        self.MODEL_TYPE = BASIC_ACTOR_CRITIC_MODEL.ContinuousBasicActorCriticEncoderSharedModel.value


class ConfigDmControlBallInCupCatchDdpg(ConfigBase, ConfigDmControlBallInCupCatch, ConfigDdpg):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigDmControlBallInCupCatch.__init__(self)
        ConfigDdpg.__init__(self)

        self.MAX_TRAINING_STEPS = 1_000_000
        self.FROM_PIXELS = True
        self.MODEL_TYPE = DDPG_MODEL.ContinuousDdpgModel.value


class ConfigDmControlBallInCupCatchTd3(ConfigBase, ConfigDmControlBallInCupCatch, ConfigTd3):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigDmControlBallInCupCatch.__init__(self)
        ConfigTd3.__init__(self)

        self.MAX_TRAINING_STEPS = 1_000_000
        self.FROM_PIXELS = True
        self.MODEL_TYPE = TD3_MODEL.ContinuousTd3EncoderModel.value


class ConfigDmControlBallInCupCatchSac(ConfigBase, ConfigDmControlBallInCupCatch, ConfigSac):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigDmControlBallInCupCatch.__init__(self)
        ConfigSac.__init__(self)

        self.MAX_TRAINING_STEPS = 1_000_000
        self.FROM_PIXELS = True
        self.MODEL_TYPE = SAC_MODEL.ContinuousSacEncoderModel.value


class ConfigDmControlBallInCupCatchTdmpc(ConfigBase, ConfigDmControlBallInCupCatch, ConfigTdmpc):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigDmControlBallInCupCatch.__init__(self)
        ConfigTdmpc.__init__(self)

        self.MAX_TRAINING_STEPS = 1_000_000
        self.FROM_PIXELS = True
        self.MODEL_TYPE = TDMPC_MODEL.TdmpcEncoderModel.value
