from link_rl.a_configuration.a_base_config.a_environments.pybullet.config_gym_pybullet import ConfigCartPoleContinuousBullet
from link_rl.a_configuration.a_base_config.b_agents.config_agents_off_policy import ConfigSac, ConfigDdpg, ConfigTd3
from link_rl.a_configuration.a_base_config.b_agents.config_agents_on_policy import ConfigA2c, ConfigPpo, ConfigPpoTrajectory, \
    ConfigReinforce, ConfigA3c
from link_rl.a_configuration.a_base_config.config_single_base import ConfigBase
from link_rl.c_models_v2.c_vanilla_policy_model import VANILLA_POLICY_MODEL
from link_rl.c_models_v2.d_basic_actor_critic_model import BASIC_ACTOR_CRITIC_MODEL
from link_rl.c_models_v2.e_ddpg_model import DDPG_MODEL
from link_rl.c_models_v2.f_td3_model import TD3_MODEL
from link_rl.c_models_v2.g_sac_model import SAC_MODEL


class ConfigCartPoleContinuousBulletReinforce(ConfigBase, ConfigCartPoleContinuousBullet, ConfigReinforce):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigCartPoleContinuousBullet.__init__(self)
        ConfigReinforce.__init__(self)

        self.MAX_TRAINING_STEPS = 100_000
        self.MODEL_TYPE = VANILLA_POLICY_MODEL.DiscreteVanillaPolicyModel.value


class ConfigCartPoleContinuousBulletA2c(ConfigBase, ConfigCartPoleContinuousBullet, ConfigA2c):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigCartPoleContinuousBullet.__init__(self)
        ConfigA2c.__init__(self)

        self.MAX_TRAINING_STEPS = 100_000
        self.MODEL_TYPE = BASIC_ACTOR_CRITIC_MODEL.ContinuousBasicActorCriticSharedModel.value


class ConfigCartPoleContinuousBulletA3c(ConfigBase, ConfigCartPoleContinuousBullet, ConfigA3c):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigCartPoleContinuousBullet.__init__(self)
        ConfigA3c.__init__(self)

        self.MAX_TRAINING_STEPS = 100_000
        self.MODEL_TYPE = BASIC_ACTOR_CRITIC_MODEL.ContinuousBasicActorCriticSharedModel.value


class ConfigCartPoleContinuousBulletPpo(ConfigBase, ConfigCartPoleContinuousBullet, ConfigPpo):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigCartPoleContinuousBullet.__init__(self)
        ConfigPpo.__init__(self)

        self.MAX_TRAINING_STEPS = 100_000
        self.MODEL_TYPE = BASIC_ACTOR_CRITIC_MODEL.ContinuousBasicActorCriticSharedModel.value


class ConfigCartPoleContinuousBulletPpoTrajectory(ConfigBase, ConfigCartPoleContinuousBullet, ConfigPpoTrajectory):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigCartPoleContinuousBullet.__init__(self)
        ConfigPpoTrajectory.__init__(self)

        self.MAX_TRAINING_STEPS = 100_000
        self.MODEL_TYPE = BASIC_ACTOR_CRITIC_MODEL.ContinuousBasicActorCriticSharedModel.value


class ConfigCartPoleContinuousBulletSac(ConfigBase, ConfigCartPoleContinuousBullet, ConfigSac):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigCartPoleContinuousBullet.__init__(self)
        ConfigSac.__init__(self)

        self.MAX_TRAINING_STEPS = 100_000
        self.MODEL_TYPE = SAC_MODEL.ContinuousSacModel.value


class ConfigCartPoleContinuousBulletDdpg(ConfigBase, ConfigCartPoleContinuousBullet, ConfigDdpg):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigCartPoleContinuousBullet.__init__(self)
        ConfigDdpg.__init__(self)

        self.MAX_TRAINING_STEPS = 200_000
        self.MODEL_TYPE = DDPG_MODEL.ContinuousDdpgModel.value


class ConfigCartPoleContinuousBulletTd3(ConfigBase, ConfigCartPoleContinuousBullet, ConfigTd3):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigCartPoleContinuousBullet.__init__(self)
        ConfigTd3.__init__(self)

        self.MAX_TRAINING_STEPS = 200_000
        self.MODEL_TYPE = TD3_MODEL.ContinuousTd3Model.value
