from link_rl.a_configuration.a_base_config.a_environments.open_ai_gym.config_gym_box2d import ConfigCarRacing
from link_rl.a_configuration.a_base_config.b_agents.config_agents_off_policy import ConfigDdpg, ConfigSac, ConfigTd3
from link_rl.a_configuration.a_base_config.b_agents.config_agents_on_policy import ConfigA2c, ConfigPpo, ConfigPpoTrajectory, ConfigA3c
from link_rl.a_configuration.a_base_config.config_single_base import ConfigBase
from link_rl.c_models_v2.d_basic_actor_critic_model import BASIC_ACTOR_CRITIC_MODEL
from link_rl.c_models_v2.e_ddpg_model import DDPG_MODEL
from link_rl.c_models_v2.f_td3_model import TD3_MODEL
from link_rl.c_models_v2.g_sac_model import SAC_MODEL


class ConfigCarRacingA2c(ConfigBase, ConfigCarRacing, ConfigA2c):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigCarRacing.__init__(self)
        ConfigA2c.__init__(self)

        self.MAX_TRAINING_STEPS = 200_000
        self.MODEL_TYPE = BASIC_ACTOR_CRITIC_MODEL.ContinuousBasicActorCriticEncoderSharedModel.value


class ConfigCarRacingA3c(ConfigBase, ConfigCarRacing, ConfigA3c):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigCarRacing.__init__(self)
        ConfigA3c.__init__(self)

        self.MAX_TRAINING_STEPS = 200_000
        self.MODEL_TYPE = BASIC_ACTOR_CRITIC_MODEL.ContinuousBasicActorCriticEncoderSharedModel.value


class ConfigCarRacingPpo(ConfigBase, ConfigCarRacing, ConfigPpo):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigCarRacing.__init__(self)
        ConfigPpo.__init__(self)

        self.MAX_TRAINING_STEPS = 200_000
        self.MODEL_TYPE = BASIC_ACTOR_CRITIC_MODEL.ContinuousBasicActorCriticEncoderSharedModel.value
        self.USE_GAE = True


class ConfigCarRacingPpoTrajectory(ConfigBase, ConfigCarRacing, ConfigPpoTrajectory):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigCarRacing.__init__(self)
        ConfigPpoTrajectory.__init__(self)

        self.MAX_TRAINING_STEPS = 500_000
        self.MODEL_TYPE = BASIC_ACTOR_CRITIC_MODEL.ContinuousBasicActorCriticEncoderSharedModel.value


class ConfigCarRacingDdpg(ConfigBase, ConfigCarRacing, ConfigDdpg):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigCarRacing.__init__(self)
        ConfigDdpg.__init__(self)

        self.MAX_TRAINING_STEPS = 300_000
        self.BUFFER_CAPACITY = 200_000
        self.MODEL_TYPE = DDPG_MODEL.ContinuousDdpgModel.value


class ConfigCarRacingTd3(ConfigBase, ConfigCarRacing, ConfigTd3):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigCarRacing.__init__(self)
        ConfigTd3.__init__(self)

        self.MAX_TRAINING_STEPS = 300_000
        self.BUFFER_CAPACITY = 200_000
        self.MODEL_TYPE = TD3_MODEL.ContinuousTd3EncoderModel.value


class ConfigCarRacingSac(ConfigBase, ConfigCarRacing, ConfigSac):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigCarRacing.__init__(self)
        ConfigSac.__init__(self)

        self.MAX_TRAINING_STEPS = 300_000
        self.BUFFER_CAPACITY = 200_000
        self.MODEL_TYPE = SAC_MODEL.ContinuousSacEncoderModel.value
