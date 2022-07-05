from link_rl.a_configuration.a_base_config.b_agents.config_agents_off_policy import ConfigDqn, ConfigDoubleDqn, ConfigDuelingDqn, \
    ConfigDoubleDuelingDqn
from link_rl.a_configuration.a_base_config.a_environments.open_ai_gym.config_gym_atari import ConfigPong
from link_rl.a_configuration.a_base_config.b_agents.config_agents_on_policy import ConfigA2c, ConfigA3c, ConfigPpo, ConfigPpoTrajectory
from link_rl.a_configuration.a_base_config.config_single_base import ConfigBase
from link_rl.c_models_v2.b_q_model import Q_MODEL
from link_rl.c_models_v2.d_basic_actor_critic_model import BASIC_ACTOR_CRITIC_MODEL


class ConfigPongDqn(ConfigBase, ConfigPong, ConfigDqn):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigPong.__init__(self)
        ConfigDqn.__init__(self)

        self.MAX_TRAINING_STEPS = 1_000_000
        self.BUFFER_CAPACITY = 250_000
        # self.MODEL_TYPE = ModelType.MEDIUM_2D_CONVOLUTIONAL
        self.MODEL_TYPE = Q_MODEL.GymAtariQModel.value


class ConfigPongDoubleDqn(ConfigBase, ConfigPong, ConfigDoubleDqn):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigPong.__init__(self)
        ConfigDoubleDqn.__init__(self)

        self.MAX_TRAINING_STEPS = 1_000_000
        self.BUFFER_CAPACITY = 250_000
        # self.MODEL_TYPE = ModelType.MEDIUM_2D_CONVOLUTIONAL
        self.MODEL_TYPE = Q_MODEL.GymAtariQModel.value


class ConfigPongDuelingDqn(ConfigBase, ConfigPong, ConfigDuelingDqn):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigPong.__init__(self)
        ConfigDuelingDqn.__init__(self)

        self.MAX_TRAINING_STEPS = 1_000_000
        self.BUFFER_CAPACITY = 250_000
        # self.MODEL_TYPE = ModelType.MEDIUM_2D_CONVOLUTIONAL
        self.MODEL_TYPE = Q_MODEL.GymAtariQModel.value


class ConfigPongDoubleDuelingDqn(ConfigBase, ConfigPong, ConfigDoubleDuelingDqn):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigPong.__init__(self)
        ConfigDoubleDuelingDqn.__init__(self)

        self.MAX_TRAINING_STEPS = 1_000_000
        self.BUFFER_CAPACITY = 250_000
        # self.MODEL_TYPE = ModelType.MEDIUM_2D_CONVOLUTIONAL
        self.MODEL_TYPE = Q_MODEL.GymAtariQModel.value


class ConfigPongA2c(ConfigBase, ConfigPong, ConfigA2c):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigPong.__init__(self)
        ConfigA2c.__init__(self)

        self.MAX_TRAINING_STEPS = 1_000_000
        # self.MODEL_TYPE = ModelType.MEDIUM_2D_CONVOLUTIONAL
        self.MODEL_TYPE = BASIC_ACTOR_CRITIC_MODEL.DiscreteBasicActorCriticEncoderSharedModel.value


class ConfigPongA3c(ConfigBase, ConfigPong, ConfigA3c):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigPong.__init__(self)
        ConfigA3c.__init__(self)

        self.MAX_TRAINING_STEPS = 1_000_000
        # self.MODEL_TYPE = ModelType.MEDIUM_2D_CONVOLUTIONAL
        self.MODEL_TYPE = Q_MODEL.GymAtariQModel.value


class ConfigPongPpo(ConfigBase, ConfigPong, ConfigPpo):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigPong.__init__(self)
        ConfigPpo.__init__(self)

        self.MAX_TRAINING_STEPS = 1_000_000
        # self.MODEL_TYPE = ModelType.MEDIUM_2D_CONVOLUTIONAL
        self.MODEL_TYPE = Q_MODEL.GymAtariQModel.value


class ConfigPongPpoTrajectory(ConfigBase, ConfigPong, ConfigPpoTrajectory):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigPong.__init__(self)
        ConfigPpoTrajectory.__init__(self)

        self.MAX_TRAINING_STEPS = 1_000_000
        # self.MODEL_TYPE = ModelType.MEDIUM_2D_CONVOLUTIONAL
        self.MODEL_TYPE = Q_MODEL.GymAtariQModel.value
