from link_rl.a_configuration.a_base_config.a_environments.dm_control.config_dm_control_cartpole import \
    ConfigDmControlCartpoleThreePoles
from link_rl.a_configuration.a_base_config.b_agents.config_agents_off_policy import ConfigDdpg, ConfigTd3, ConfigSac, \
    ConfigTdmpc
from link_rl.a_configuration.a_base_config.b_agents.config_agents_on_policy import ConfigA2c, ConfigPpo, ConfigA3c
from link_rl.a_configuration.a_base_config.config_single_base import ConfigBase
from link_rl.c_models_v2.d_basic_actor_critic_model import BASIC_ACTOR_CRITIC_MODEL
from link_rl.c_models_v2.e_ddpg_model import DDPG_MODEL
from link_rl.c_models_v2.f_td3_model import TD3_MODEL
from link_rl.c_models_v2.g_sac_model import SAC_MODEL


class ConfigDmControlCartPoleThreePolesA2c(ConfigBase, ConfigDmControlCartpoleThreePoles, ConfigA2c):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigDmControlCartpoleThreePoles.__init__(self)
        ConfigA2c.__init__(self)

        self.MAX_TRAINING_STEPS = 500_000
        self.MODEL_TYPE = BASIC_ACTOR_CRITIC_MODEL.ContinuousBasicActorCriticSharedModel.value


class ConfigDmControlCartPoleThreePolesA3c(ConfigBase, ConfigDmControlCartpoleThreePoles, ConfigA3c):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigDmControlCartpoleThreePoles.__init__(self)
        ConfigA3c.__init__(self)

        self.MAX_TRAINING_STEPS = 500_000
        self.MODEL_TYPE = BASIC_ACTOR_CRITIC_MODEL.ContinuousBasicActorCriticSharedModel.value


class ConfigDmControlCartPoleThreePolesPpo(ConfigBase, ConfigDmControlCartpoleThreePoles, ConfigPpo):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigDmControlCartpoleThreePoles.__init__(self)
        ConfigPpo.__init__(self)

        self.MAX_TRAINING_STEPS = 500_000
        self.MODEL_TYPE = BASIC_ACTOR_CRITIC_MODEL.ContinuousBasicActorCriticSharedModel.value


class ConfigDmControlCartPoleThreePolesDdpg(ConfigBase, ConfigDmControlCartpoleThreePoles, ConfigDdpg):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigDmControlCartpoleThreePoles.__init__(self)
        ConfigDdpg.__init__(self)

        self.MAX_TRAINING_STEPS = 500_000
        self.MODEL_TYPE = DDPG_MODEL.ContinuousDdpgModel.value


class ConfigDmControlCartPoleThreePolesTd3(ConfigBase, ConfigDmControlCartpoleThreePoles, ConfigTd3):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigDmControlCartpoleThreePoles.__init__(self)
        ConfigTd3.__init__(self)

        self.MAX_TRAINING_STEPS = 500_000
        self.MODEL_TYPE = TD3_MODEL.ContinuousTd3Model.value


class ConfigDmControlCartPoleThreePolesSac(ConfigBase, ConfigDmControlCartpoleThreePoles, ConfigSac):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigDmControlCartpoleThreePoles.__init__(self)
        ConfigSac.__init__(self)

        self.MAX_TRAINING_STEPS = 500_000
        self.MODEL_TYPE = SAC_MODEL.ContinuousSacModel.value


class ConfigDmControlCartPoleThreePolesTdmpc(ConfigBase, ConfigDmControlCartpoleThreePoles, ConfigTdmpc):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigDmControlCartpoleThreePoles.__init__(self)
        ConfigTdmpc.__init__(self)

        self.MAX_TRAINING_STEPS = 500_000
        self.MODEL_TYPE = TDMPC_MODEL.TdmpcModel.value
