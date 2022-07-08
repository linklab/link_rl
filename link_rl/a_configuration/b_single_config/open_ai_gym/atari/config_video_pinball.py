from link_rl.a_configuration.a_base_config.b_agents.config_agents_off_policy import ConfigDqn, ConfigDoubleDqn, ConfigDuelingDqn, \
    ConfigDoubleDuelingDqn
from link_rl.a_configuration.a_base_config.a_environments.open_ai_gym.config_gym_atari import ConfigVideoPinball
from link_rl.a_configuration.a_base_config.b_agents.config_agents_on_policy import ConfigA2c, ConfigA3c, ConfigPpo, ConfigPpoTrajectory
from link_rl.a_configuration.a_base_config.config_single_base import ConfigBase
from link_rl.c_models_v2.b_q_model import Q_MODEL
from link_rl.c_models_v2.d_basic_actor_critic_model import BASIC_ACTOR_CRITIC_MODEL


class ConfigVideoPinballDqn(ConfigBase, ConfigVideoPinball, ConfigDqn):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigVideoPinball.__init__(self)
        ConfigDqn.__init__(self)

        self.MAX_TRAINING_STEPS = 1_000_000
        self.BUFFER_CAPACITY = 500_000
        self.MODEL_TYPE = Q_MODEL.GymAtariQModel.value


class ConfigVideoPinballDoubleDqn(ConfigBase, ConfigVideoPinball, ConfigDoubleDqn):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigVideoPinball.__init__(self)
        ConfigDoubleDqn.__init__(self)

        self.MAX_TRAINING_STEPS = 1_000_000
        self.BUFFER_CAPACITY = 500_000
        self.MODEL_TYPE = Q_MODEL.GymAtariQModel.value


class ConfigVideoPinballDuelingDqn(ConfigBase, ConfigVideoPinball, ConfigDuelingDqn):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigVideoPinball.__init__(self)
        ConfigDuelingDqn.__init__(self)

        self.MAX_TRAINING_STEPS = 1_000_000
        self.BUFFER_CAPACITY = 500_000
        self.MODEL_TYPE = Q_MODEL.GymAtariQModel.value


class ConfigVideoPinballDoubleDuelingDqn(ConfigBase, ConfigVideoPinball, ConfigDoubleDuelingDqn):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigVideoPinball.__init__(self)
        ConfigDoubleDuelingDqn.__init__(self)

        self.MAX_TRAINING_STEPS = 1_000_000
        self.BUFFER_CAPACITY = 500_000
        self.MODEL_TYPE = Q_MODEL.GymAtariQModel.value


class ConfigVideoPinballA2c(ConfigBase, ConfigVideoPinball, ConfigA2c):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigVideoPinball.__init__(self)
        ConfigA2c.__init__(self)

        self.MAX_TRAINING_STEPS = 1_000_000
        self.MODEL_TYPE = BASIC_ACTOR_CRITIC_MODEL.DiscreteBasicActorCriticEncoderSharedModel.value


class ConfigVideoPinballA3c(ConfigBase, ConfigVideoPinball, ConfigA3c):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigVideoPinball.__init__(self)
        ConfigA3c.__init__(self)

        self.MAX_TRAINING_STEPS = 1_000_000
        self.MODEL_TYPE = BASIC_ACTOR_CRITIC_MODEL.DiscreteBasicActorCriticEncoderSharedModel.value


class ConfigVideoPinballPpo(ConfigBase, ConfigVideoPinball, ConfigPpo):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigVideoPinball.__init__(self)
        ConfigPpo.__init__(self)

        self.MAX_TRAINING_STEPS = 1_000_000
        self.MODEL_TYPE = BASIC_ACTOR_CRITIC_MODEL.DiscreteBasicActorCriticEncoderSharedModel.value


class ConfigVideoPinballPpoTrajectory(ConfigBase, ConfigVideoPinball, ConfigPpoTrajectory):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigVideoPinball.__init__(self)
        ConfigPpoTrajectory.__init__(self)

        self.MAX_TRAINING_STEPS = 1_000_000
        self.MODEL_TYPE = BASIC_ACTOR_CRITIC_MODEL.DiscreteBasicActorCriticEncoderSharedModel.value
