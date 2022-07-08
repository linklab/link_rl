from link_rl.a_configuration.a_base_config.a_environments.open_ai_gym.config_gym_box2d import ConfigLunarLanderContinuous
from link_rl.a_configuration.a_base_config.b_agents.config_agents_off_policy import ConfigDdpg, ConfigSac, ConfigTd3
from link_rl.a_configuration.a_base_config.b_agents.config_agents_on_policy import ConfigA2c, ConfigPpo, ConfigPpoTrajectory, \
    ConfigA3c, ConfigAsynchronousPpo
from link_rl.a_configuration.a_base_config.config_single_base import ConfigBase
from link_rl.c_models_v2.d_basic_actor_critic_model import BASIC_ACTOR_CRITIC_MODEL
from link_rl.c_models_v2.e_ddpg_model import DDPG_MODEL
from link_rl.c_models_v2.f_td3_model import TD3_MODEL
from link_rl.c_models_v2.g_sac_model import SAC_MODEL


class ConfigLunarLanderContinuousA2c(ConfigBase, ConfigLunarLanderContinuous, ConfigA2c):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigLunarLanderContinuous.__init__(self)
        ConfigA2c.__init__(self)

        self.MAX_TRAINING_STEPS = 200_000
        self.MODEL_TYPE = BASIC_ACTOR_CRITIC_MODEL.ContinuousBasicActorCriticSharedModel.value


class ConfigLunarLanderContinuousA3c(ConfigBase, ConfigLunarLanderContinuous, ConfigA3c):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigLunarLanderContinuous.__init__(self)
        ConfigA3c.__init__(self)

        self.MAX_TRAINING_STEPS = 200_000
        self.MODEL_TYPE = BASIC_ACTOR_CRITIC_MODEL.ContinuousBasicActorCriticSharedModel.value


class ConfigLunarLanderContinuousPpo(ConfigBase, ConfigLunarLanderContinuous, ConfigPpo):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigLunarLanderContinuous.__init__(self)
        ConfigPpo.__init__(self)

        self.MAX_TRAINING_STEPS = 200_000
        self.MODEL_TYPE = BASIC_ACTOR_CRITIC_MODEL.ContinuousBasicActorCriticSharedModel.value
        self.USE_GAE = True


class ConfigLunarLanderContinuousAsynchronousPpo(ConfigBase, ConfigLunarLanderContinuous, ConfigAsynchronousPpo):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigLunarLanderContinuous.__init__(self)
        ConfigAsynchronousPpo.__init__(self)

        self.MAX_TRAINING_STEPS = 200_000
        self.MODEL_TYPE = BASIC_ACTOR_CRITIC_MODEL.ContinuousBasicActorCriticSharedModel.value
        self.USE_GAE = True


class ConfigLunarLanderContinuousPpoTrajectory(ConfigBase, ConfigLunarLanderContinuous, ConfigPpoTrajectory):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigLunarLanderContinuous.__init__(self)
        ConfigPpoTrajectory.__init__(self)

        self.MAX_TRAINING_STEPS = 500_000
        self.MODEL_TYPE = BASIC_ACTOR_CRITIC_MODEL.ContinuousBasicActorCriticSharedModel.value


class ConfigLunarLanderContinuousDdpg(ConfigBase, ConfigLunarLanderContinuous, ConfigDdpg):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigLunarLanderContinuous.__init__(self)
        ConfigDdpg.__init__(self)

        self.MAX_TRAINING_STEPS = 300_000
        self.BUFFER_CAPACITY = 200_000
        self.MODEL_TYPE = DDPG_MODEL.ContinuousDdpgModel.value


class ConfigLunarLanderContinuousTd3(ConfigBase, ConfigLunarLanderContinuous, ConfigTd3):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigLunarLanderContinuous.__init__(self)
        ConfigTd3.__init__(self)

        self.MAX_TRAINING_STEPS = 300_000
        self.BUFFER_CAPACITY = 200_000
        self.MODEL_TYPE = TD3_MODEL.ContinuousTd3Model.value


class ConfigLunarLanderContinuousSac(ConfigBase, ConfigLunarLanderContinuous, ConfigSac):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigLunarLanderContinuous.__init__(self)
        ConfigSac.__init__(self)

        self.MAX_TRAINING_STEPS = 300_000
        self.BUFFER_CAPACITY = 200_000
        self.MODEL_TYPE = SAC_MODEL.ContinuousSacModel.value
