from link_rl.a_configuration.a_base_config.a_environments.ai_birds.config_ai_birds import ConfigAiBirds
from link_rl.a_configuration.a_base_config.b_agents.config_agents_off_policy import ConfigDqn, ConfigDoubleDqn, \
    ConfigDuelingDqn, ConfigDoubleDuelingDqn
from link_rl.a_configuration.a_base_config.b_agents.config_agents_on_policy import ConfigA2c, ConfigPpo
from link_rl.a_configuration.a_base_config.config_single_base import ConfigBase
from link_rl.c_models_v2.b_q_model import Q_MODEL
from link_rl.c_models_v2.d_basic_actor_critic_model import BASIC_ACTOR_CRITIC_MODEL


class ConfigAiBirdsDqn(ConfigBase, ConfigAiBirds, ConfigDqn):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigAiBirds.__init__(self)
        ConfigDqn.__init__(self)

        self.LEARNING_RATE = 0.001
        self.MAX_TRAINING_STEPS = 1_000_000
        self.BUFFER_CAPACITY = 500_000
        self.MODEL_TYPE = Q_MODEL.QModel

        self.BATCH_SIZE = 128
        self.MIN_BUFFER_SIZE_FOR_TRAIN = self.BATCH_SIZE
        self.CONSOLE_LOG_INTERVAL_TRAINING_STEPS = 3
        self.TEST_INTERVAL_TRAINING_STEPS = 10


class ConfigAiBirdsDoubleDqn(ConfigBase, ConfigAiBirds, ConfigDoubleDqn):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigAiBirds.__init__(self)
        ConfigDoubleDqn.__init__(self)

        self.LEARNING_RATE = 0.001
        self.MAX_TRAINING_STEPS = 1_000_000
        self.BUFFER_CAPACITY = 500_000
        self.MODEL_TYPE = Q_MODEL.QModel

        self.BATCH_SIZE = 128
        self.MIN_BUFFER_SIZE_FOR_TRAIN = self.BATCH_SIZE
        self.CONSOLE_LOG_INTERVAL_TRAINING_STEPS = 3
        self.TEST_INTERVAL_TRAINING_STEPS = 10


class ConfigAiBirdsDuelingDqn(ConfigBase, ConfigAiBirds, ConfigDuelingDqn):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigAiBirds.__init__(self)
        ConfigDuelingDqn.__init__(self)

        self.LEARNING_RATE = 0.001
        self.MAX_TRAINING_STEPS = 1_000_000
        self.BUFFER_CAPACITY = 500_000
        self.MODEL_TYPE = Q_MODEL.DuelingQModel.value

        self.BATCH_SIZE = 128
        self.MIN_BUFFER_SIZE_FOR_TRAIN = self.BATCH_SIZE
        self.CONSOLE_LOG_INTERVAL_TRAINING_STEPS = 3
        self.TEST_INTERVAL_TRAINING_STEPS = 10


class ConfigAiBirdsDoubleDuelingDqn(ConfigBase, ConfigAiBirds, ConfigDoubleDuelingDqn):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigAiBirds.__init__(self)
        ConfigDoubleDuelingDqn.__init__(self)

        self.LEARNING_RATE = 0.001
        self.MAX_TRAINING_STEPS = 1_000_000
        self.BUFFER_CAPACITY = 500_000
        self.MODEL_TYPE = Q_MODEL.DuelingQModel.value

        self.BATCH_SIZE = 128
        self.MIN_BUFFER_SIZE_FOR_TRAIN = self.BATCH_SIZE
        self.CONSOLE_LOG_INTERVAL_TRAINING_STEPS = 3
        self.TEST_INTERVAL_TRAINING_STEPS = 10


# OnPolicy
class ConfigAiBirdsA2c(ConfigBase, ConfigAiBirds, ConfigA2c):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigAiBirds.__init__(self)
        ConfigA2c.__init__(self)

        self.MAX_TRAINING_STEPS = 1_000_000
        self.MODEL_TYPE = BASIC_ACTOR_CRITIC_MODEL.DiscreteBasicActorCriticSharedModel.value

        self.CONSOLE_LOG_INTERVAL_TRAINING_STEPS = 1
        self.TEST_INTERVAL_TRAINING_STEPS = 10


class ConfigAiBirdsPpo(ConfigBase, ConfigAiBirds, ConfigPpo):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigAiBirds.__init__(self)
        ConfigPpo.__init__(self)

        self.MAX_TRAINING_STEPS = 1_000_000
        self.MODEL_TYPE = BASIC_ACTOR_CRITIC_MODEL.DiscreteBasicActorCriticSharedModel.value

        self.CONSOLE_LOG_INTERVAL_TRAINING_STEPS = self.PPO_K_EPOCH
        self.TEST_INTERVAL_TRAINING_STEPS = 10
        self.BATCH_SIZE = 128
