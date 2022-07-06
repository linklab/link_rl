from link_rl.a_configuration.a_base_config.b_agents.config_agents_off_policy import ConfigDqn, ConfigDoubleDqn, ConfigDuelingDqn, \
    ConfigDoubleDuelingDqn
from link_rl.a_configuration.a_base_config.a_environments.open_ai_gym.config_gym_atari import ConfigDemonAttack
from link_rl.a_configuration.a_base_config.b_agents.config_agents_on_policy import ConfigA2c, ConfigA3c, ConfigPpo, ConfigPpoTrajectory
from link_rl.a_configuration.a_base_config.config_single_base import ConfigBase
from link_rl.c_models_v2.b_q_model import Q_MODEL
from link_rl.c_models_v2.d_basic_actor_critic_model import BASIC_ACTOR_CRITIC_MODEL
from link_rl.g_utils.types import ModelType


class ConfigDemonAttackDqn(ConfigBase, ConfigDemonAttack, ConfigDqn):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigDemonAttack.__init__(self)
        ConfigDqn.__init__(self)

        self.MAX_TRAINING_STEPS = 1_000_000
        self.BUFFER_CAPACITY = 1_000_000
        self.MODEL_TYPE = Q_MODEL.GymAtariQModel.value


class ConfigDemonAttackDoubleDqn(ConfigBase, ConfigDemonAttack, ConfigDoubleDqn):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigDemonAttack.__init__(self)
        ConfigDoubleDqn.__init__(self)

        self.MAX_TRAINING_STEPS = 1_000_000
        self.BUFFER_CAPACITY = 1_000_000
        self.MODEL_TYPE = Q_MODEL.GymAtariQModel.value


class ConfigDemonAttackDuelingDqn(ConfigBase, ConfigDemonAttack, ConfigDuelingDqn):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigDemonAttack.__init__(self)
        ConfigDuelingDqn.__init__(self)

        self.MAX_TRAINING_STEPS = 1_000_000
        self.BUFFER_CAPACITY = 1_000_000
        self.MODEL_TYPE = Q_MODEL.GymAtariQModel.value


class ConfigDemonAttackDoubleDuelingDqn(ConfigBase, ConfigDemonAttack, ConfigDoubleDuelingDqn):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigDemonAttack.__init__(self)
        ConfigDoubleDuelingDqn.__init__(self)

        self.MAX_TRAINING_STEPS = 1_000_000
        self.BUFFER_CAPACITY = 1_000_000
        self.MODEL_TYPE = Q_MODEL.GymAtariQModel.value


class ConfigDemonAttackA2c(ConfigBase, ConfigDemonAttack, ConfigA2c):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigDemonAttack.__init__(self)
        ConfigA2c.__init__(self)

        self.MAX_TRAINING_STEPS = 1_000_000
        self.MODEL_TYPE = BASIC_ACTOR_CRITIC_MODEL.DiscreteBasicActorCriticEncoderSharedModel.value


class ConfigDemonAttackA3c(ConfigBase, ConfigDemonAttack, ConfigA3c):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigDemonAttack.__init__(self)
        ConfigA3c.__init__(self)

        self.MAX_TRAINING_STEPS = 1_000_000
        self.MODEL_TYPE = BASIC_ACTOR_CRITIC_MODEL.DiscreteBasicActorCriticEncoderSharedModel.value


class ConfigDemonAttackPpo(ConfigBase, ConfigDemonAttack, ConfigPpo):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigDemonAttack.__init__(self)
        ConfigPpo.__init__(self)

        self.MAX_TRAINING_STEPS = 1_000_000
        self.MODEL_TYPE = BASIC_ACTOR_CRITIC_MODEL.DiscreteBasicActorCriticEncoderSharedModel.value


class ConfigDemonAttackPpoTrajectory(ConfigBase, ConfigDemonAttack, ConfigPpoTrajectory):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigDemonAttack.__init__(self)
        ConfigPpoTrajectory.__init__(self)

        self.MAX_TRAINING_STEPS = 1_000_000
        self.MODEL_TYPE = BASIC_ACTOR_CRITIC_MODEL.DiscreteBasicActorCriticEncoderSharedModel.value
