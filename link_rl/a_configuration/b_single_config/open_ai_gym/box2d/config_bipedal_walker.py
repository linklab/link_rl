from link_rl.a_configuration.a_base_config.a_environments.open_ai_gym.config_gym_box2d import ConfigNormalBipedalWalker, \
    ConfigHardcoreBipedalWalker
from link_rl.a_configuration.a_base_config.b_agents.config_agents_off_policy import ConfigDdpg, ConfigSac, ConfigTd3
from link_rl.a_configuration.a_base_config.b_agents.config_agents_on_policy import ConfigA2c, ConfigPpo, ConfigPpoTrajectory, ConfigA3c
from link_rl.a_configuration.a_base_config.config_single_base import ConfigBase
from link_rl.c_models_v2.d_basic_actor_critic_model import BASIC_ACTOR_CRITIC_MODEL
from link_rl.c_models_v2.e_ddpg_model import DDPG_MODEL
from link_rl.c_models_v2.f_td3_model import TD3_MODEL
from link_rl.c_models_v2.g_sac_model import SAC_MODEL


class ConfigNormalBipedalWalkerA2c(ConfigBase, ConfigNormalBipedalWalker, ConfigA2c):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigNormalBipedalWalker.__init__(self)
        ConfigA2c.__init__(self)

        self.MAX_TRAINING_STEPS = 200_000
        self.MODEL_TYPE = BASIC_ACTOR_CRITIC_MODEL.DiscreteBasicActorCriticSharedModel.value


class ConfigNormalBipedalWalkerA3c(ConfigBase, ConfigNormalBipedalWalker, ConfigA3c):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigNormalBipedalWalker.__init__(self)
        ConfigA3c.__init__(self)

        self.MAX_TRAINING_STEPS = 200_000
        self.MODEL_TYPE = BASIC_ACTOR_CRITIC_MODEL.DiscreteBasicActorCriticSharedModel.value


class ConfigNormalBipedalWalkerPpo(ConfigBase, ConfigNormalBipedalWalker, ConfigPpo):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigNormalBipedalWalker.__init__(self)
        ConfigPpo.__init__(self)

        self.MAX_TRAINING_STEPS = 200_000
        self.MODEL_TYPE = BASIC_ACTOR_CRITIC_MODEL.DiscreteBasicActorCriticSharedModel.value
        self.USE_GAE = True


class ConfigNormalBipedalWalkerPpoTrajectory(ConfigBase, ConfigNormalBipedalWalker, ConfigPpoTrajectory):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigNormalBipedalWalker.__init__(self)
        ConfigPpoTrajectory.__init__(self)

        self.MAX_TRAINING_STEPS = 500_000
        self.MODEL_TYPE = BASIC_ACTOR_CRITIC_MODEL.DiscreteBasicActorCriticSharedModel.value


class ConfigNormalBipedalWalkerDdpg(ConfigBase, ConfigNormalBipedalWalker, ConfigDdpg):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigNormalBipedalWalker.__init__(self)
        ConfigDdpg.__init__(self)

        self.MAX_TRAINING_STEPS = 300_000
        self.BUFFER_CAPACITY = 200_000
        self.MODEL_TYPE = DDPG_MODEL.ContinuousDdpgModel.value


class ConfigNormalBipedalWalkerTd3(ConfigBase, ConfigNormalBipedalWalker, ConfigTd3):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigNormalBipedalWalker.__init__(self)
        ConfigTd3.__init__(self)

        self.MAX_TRAINING_STEPS = 300_000
        self.BUFFER_CAPACITY = 200_000
        self.MODEL_TYPE = TD3_MODEL.ContinuousTd3Model.value


class ConfigNormalBipedalWalkerSac(ConfigBase, ConfigNormalBipedalWalker, ConfigSac):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigNormalBipedalWalker.__init__(self)
        ConfigSac.__init__(self)

        self.MAX_TRAINING_STEPS = 300_000
        self.BUFFER_CAPACITY = 200_000
        self.MODEL_TYPE = SAC_MODEL.ContinuousSacModel.value

####

class ConfigHardcoreBipedalWalkerA2c(ConfigBase, ConfigHardcoreBipedalWalker, ConfigA2c):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigHardcoreBipedalWalker.__init__(self)
        ConfigA2c.__init__(self)

        self.MAX_TRAINING_STEPS = 1_000_000
        self.MODEL_TYPE = BASIC_ACTOR_CRITIC_MODEL.DiscreteBasicActorCriticSharedModel.value


class ConfigHardcoreBipedalWalkerA3c(ConfigBase, ConfigHardcoreBipedalWalker, ConfigA3c):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigHardcoreBipedalWalker.__init__(self)
        ConfigA3c.__init__(self)

        self.MAX_TRAINING_STEPS = 1_000_000
        self.MODEL_TYPE = BASIC_ACTOR_CRITIC_MODEL.DiscreteBasicActorCriticSharedModel.value


class ConfigHardcoreBipedalWalkerPpo(ConfigBase, ConfigHardcoreBipedalWalker, ConfigPpo):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigHardcoreBipedalWalker.__init__(self)
        ConfigPpo.__init__(self)

        self.MAX_TRAINING_STEPS = 1_000_000
        self.MODEL_TYPE = BASIC_ACTOR_CRITIC_MODEL.DiscreteBasicActorCriticSharedModel.value

        self.USE_GAE = True


class ConfigHardcoreBipedalWalkerPpoTrajectory(ConfigBase, ConfigHardcoreBipedalWalker, ConfigPpoTrajectory):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigHardcoreBipedalWalker.__init__(self)
        ConfigPpoTrajectory.__init__(self)

        self.MAX_TRAINING_STEPS = 1_000_000
        self.MODEL_TYPE = BASIC_ACTOR_CRITIC_MODEL.DiscreteBasicActorCriticSharedModel.value


class ConfigHardcoreBipedalWalkerDdpg(ConfigBase, ConfigHardcoreBipedalWalker, ConfigDdpg):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigHardcoreBipedalWalker.__init__(self)
        ConfigDdpg.__init__(self)

        self.MAX_TRAINING_STEPS = 1_000_000
        self.BUFFER_CAPACITY = 500_000
        self.MODEL_TYPE = DDPG_MODEL.ContinuousDdpgModel.value


class ConfigHardcoreBipedalWalkerTd3(ConfigBase, ConfigHardcoreBipedalWalker, ConfigTd3):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigHardcoreBipedalWalker.__init__(self)
        ConfigTd3.__init__(self)

        self.MAX_TRAINING_STEPS = 1_000_000
        self.BUFFER_CAPACITY = 500_000
        self.MODEL_TYPE = TD3_MODEL.ContinuousTd3Model.value


class ConfigHardcoreBipedalWalkerSac(ConfigBase, ConfigHardcoreBipedalWalker, ConfigSac):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigHardcoreBipedalWalker.__init__(self)
        ConfigSac.__init__(self)

        self.MAX_TRAINING_STEPS = 1_000_000
        self.BUFFER_CAPACITY = 500_000
        self.MODEL_TYPE = SAC_MODEL.ContinuousSacModel.value
        self.MIN_ALPHA = 0.25
