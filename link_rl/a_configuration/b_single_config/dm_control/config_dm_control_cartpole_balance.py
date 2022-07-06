from link_rl.a_configuration.a_base_config.a_environments.dm_control.config_dm_control_cartpole import \
    ConfigDmControlCartpoleBalance
from link_rl.a_configuration.a_base_config.b_agents.config_agents_off_policy import ConfigDdpg, ConfigTd3, ConfigSac, \
    ConfigTdmpc
from link_rl.a_configuration.a_base_config.b_agents.config_agents_on_policy import ConfigA2c, ConfigPpo, ConfigA3c, \
    ConfigAsynchronousPpo
from link_rl.a_configuration.a_base_config.config_single_base import ConfigBase
from link_rl.c_models_v2.d_basic_actor_critic_model import BASIC_ACTOR_CRITIC_MODEL
from link_rl.c_models_v2.e_ddpg_model import DDPG_MODEL
from link_rl.c_models_v2.f_td3_model import TD3_MODEL
from link_rl.c_models_v2.g_sac_model import SAC_MODEL
from link_rl.c_models_v2.h_tdmpc_model import TDMPC_MODEL
from link_rl.g_utils.types import ModelType


class ConfigDmControlCartPoleBalanceA2c(ConfigBase, ConfigDmControlCartpoleBalance, ConfigA2c):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigDmControlCartpoleBalance.__init__(self)
        ConfigA2c.__init__(self)

        self.MAX_TRAINING_STEPS = 300_000
        #self.MODEL_TYPE = ModelType.SMALL_LINEAR
        self.FROM_PIXELS = True
        self.MODEL_TYPE = BASIC_ACTOR_CRITIC_MODEL.ContinuousBasicActorCriticEncoderSharedModel.value


class ConfigDmControlCartPoleBalanceA3c(ConfigBase, ConfigDmControlCartpoleBalance, ConfigA3c):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigDmControlCartpoleBalance.__init__(self)
        ConfigA3c.__init__(self)

        self.MAX_TRAINING_STEPS = 300_000
        #self.MODEL_TYPE = ModelType.SMALL_LINEAR
        self.FROM_PIXELS = True
        self.MODEL_TYPE = BASIC_ACTOR_CRITIC_MODEL.ContinuousBasicActorCriticEncoderSharedModel.value


class ConfigDmControlCartPoleBalancePpo(ConfigBase, ConfigDmControlCartpoleBalance, ConfigPpo):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigDmControlCartpoleBalance.__init__(self)
        ConfigPpo.__init__(self)

        self.MAX_TRAINING_STEPS = 300_000
        #self.MODEL_TYPE = ModelType.SMALL_LINEAR
        self.FROM_PIXELS = True
        self.MODEL_TYPE = BASIC_ACTOR_CRITIC_MODEL.ContinuousBasicActorCriticEncoderSharedModel.value


class ConfigDmControlCartPoleBalanceAsynchronousPpo(ConfigBase, ConfigDmControlCartpoleBalance, ConfigAsynchronousPpo):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigDmControlCartpoleBalance.__init__(self)
        ConfigAsynchronousPpo.__init__(self)

        self.MAX_TRAINING_STEPS = 300_000
        # self.MODEL_TYPE = ModelType.SMALL_LINEAR
        self.FROM_PIXELS = True
        self.MODEL_TYPE = BASIC_ACTOR_CRITIC_MODEL.ContinuousBasicActorCriticEncoderSharedModel.value


class ConfigDmControlCartPoleBalanceDdpg(ConfigBase, ConfigDmControlCartpoleBalance, ConfigDdpg):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigDmControlCartpoleBalance.__init__(self)
        ConfigDdpg.__init__(self)

        self.MAX_TRAINING_STEPS = 300_000
        #self.MODEL_TYPE = ModelType.SMALL_LINEAR
        self.FROM_PIXELS = True
        #self.MODEL_TYPE = DDPG_MODEL.ContinuousDdpgModel.value


class ConfigDmControlCartPoleBalanceTd3(ConfigBase, ConfigDmControlCartpoleBalance, ConfigTd3):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigDmControlCartpoleBalance.__init__(self)
        ConfigTd3.__init__(self)

        self.MAX_TRAINING_STEPS = 300_000
        # self.MODEL_TYPE = ModelType.SMALL_LINEAR
        self.FROM_PIXELS = True
        self.MODEL_TYPE = TD3_MODEL.ContinuousTd3EncoderModel.value


class ConfigDmControlCartPoleBalanceSac(ConfigBase, ConfigDmControlCartpoleBalance, ConfigSac):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigDmControlCartpoleBalance.__init__(self)
        ConfigSac.__init__(self)

        self.MAX_TRAINING_STEPS = 300_000
        #self.MODEL_TYPE = ModelType.SMALL_LINEAR
        self.FROM_PIXELS = True
        self.MODEL_TYPE = SAC_MODEL.ContinuousSacEncoderModel.value


class ConfigDmControlCartPoleBalanceTdmpc(ConfigBase, ConfigDmControlCartpoleBalance, ConfigTdmpc):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigDmControlCartpoleBalance.__init__(self)
        ConfigTdmpc.__init__(self)

        self.MAX_TRAINING_STEPS = 300_000
        self.MODEL_TYPE = ModelType.SMALL_LINEAR
        self.FROM_PIXELS = True
        self.MODEL_TYPE = TDMPC_MODEL.TdmpcEncoderModel.value