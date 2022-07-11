from link_rl.a_configuration.a_base_config.b_agents.config_agents_off_policy import ConfigDqn, ConfigDoubleDqn, ConfigDuelingDqn, \
    ConfigDoubleDuelingDqn
from link_rl.a_configuration.a_base_config.a_environments.open_ai_gym.config_gym_atari import ConfigBreakout
from link_rl.a_configuration.a_base_config.b_agents.config_agents_on_policy import ConfigA2c, ConfigA3c, ConfigPpo, ConfigPpoTrajectory
from link_rl.a_configuration.a_base_config.config_single_base import ConfigBase
from link_rl.d_models.b_q_model import Q_MODEL
from link_rl.d_models.d_basic_actor_critic_model import BASIC_ACTOR_CRITIC_MODEL


class ConfigBreakoutDqn(ConfigBase, ConfigBreakout, ConfigDqn):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigBreakout.__init__(self)
        ConfigDqn.__init__(self)

        self.MAX_TRAINING_STEPS = 1_000_000
        self.BUFFER_CAPACITY = 500_000
        self.MODEL_TYPE = Q_MODEL.QModel.value


class ConfigBreakoutDoubleDqn(ConfigBase, ConfigBreakout, ConfigDoubleDqn):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigBreakout.__init__(self)
        ConfigDoubleDqn.__init__(self)

        self.MAX_TRAINING_STEPS = 1_000_000
        self.BUFFER_CAPACITY = 500_000
        self.MODEL_TYPE = Q_MODEL.QModel.value


class ConfigBreakoutDuelingDqn(ConfigBase, ConfigBreakout, ConfigDuelingDqn):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigBreakout.__init__(self)
        ConfigDuelingDqn.__init__(self)

        self.MAX_TRAINING_STEPS = 1_000_000
        self.BUFFER_CAPACITY = 500_000
        self.MODEL_TYPE = Q_MODEL.DuelingQModel.value


class ConfigBreakoutDoubleDuelingDqn(ConfigBase, ConfigBreakout, ConfigDoubleDuelingDqn):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigBreakout.__init__(self)
        ConfigDoubleDuelingDqn.__init__(self)

        self.MAX_TRAINING_STEPS = 1_000_000
        self.BUFFER_CAPACITY = 500_000
        self.MODEL_TYPE = Q_MODEL.DuelingQModel.value


class ConfigBreakoutA2c(ConfigBase, ConfigBreakout, ConfigA2c):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigBreakout.__init__(self)
        ConfigA2c.__init__(self)

        self.MAX_TRAINING_STEPS = 1_000_000
        self.MODEL_TYPE = BASIC_ACTOR_CRITIC_MODEL.DiscreteBasicActorCriticSharedModel.value


class ConfigBreakoutA3c(ConfigBase, ConfigBreakout, ConfigA3c):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigBreakout.__init__(self)
        ConfigA3c.__init__(self)

        self.MAX_TRAINING_STEPS = 1_000_000
        self.MODEL_TYPE = BASIC_ACTOR_CRITIC_MODEL.DiscreteBasicActorCriticSharedModel.value


class ConfigBreakoutPpo(ConfigBase, ConfigBreakout, ConfigPpo):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigBreakout.__init__(self)
        ConfigPpo.__init__(self)

        self.MAX_TRAINING_STEPS = 1_000_000
        self.MODEL_TYPE = BASIC_ACTOR_CRITIC_MODEL.DiscreteBasicActorCriticSharedModel.value


class ConfigBreakoutPpoTrajectory(ConfigBase, ConfigBreakout, ConfigPpoTrajectory):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigBreakout.__init__(self)
        ConfigPpoTrajectory.__init__(self)

        self.MAX_TRAINING_STEPS = 1_000_000
        self.MODEL_TYPE = BASIC_ACTOR_CRITIC_MODEL.DiscreteBasicActorCriticSharedModel.value
