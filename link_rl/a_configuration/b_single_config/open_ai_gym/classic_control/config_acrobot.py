from link_rl.a_configuration.a_base_config.b_agents.config_agents_off_policy import ConfigDqn, ConfigDoubleDqn, ConfigDuelingDqn, \
    ConfigDoubleDuelingDqn
from link_rl.a_configuration.a_base_config.b_agents.config_agents_on_policy import ConfigA2c, ConfigReinforce, ConfigPpo, \
    ConfigPpoTrajectory, ConfigA3c
from link_rl.a_configuration.a_base_config.config_single_base import ConfigBase
from link_rl.a_configuration.a_base_config.a_environments.open_ai_gym.config_gym_classic_control import ConfigAcrobot
from link_rl.d_models.b_q_model import Q_MODEL
from link_rl.d_models.c_vanilla_policy_model import VANILLA_POLICY_MODEL
from link_rl.d_models.d_basic_actor_critic_model import BASIC_ACTOR_CRITIC_MODEL


class ConfigAcrobotDqn(ConfigBase, ConfigAcrobot, ConfigDqn):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigAcrobot.__init__(self)
        ConfigDqn.__init__(self)

        self.LEARNING_RATE = 0.001
        self.MAX_TRAINING_STEPS = 100_000
        self.BUFFER_CAPACITY = 50_000
        self.MODEL_TYPE = Q_MODEL.QModel.value


class ConfigAcrobotDoubleDqn(ConfigBase, ConfigAcrobot, ConfigDoubleDqn):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigAcrobot.__init__(self)
        ConfigDoubleDqn.__init__(self)

        self.LEARNING_RATE = 0.001
        self.MAX_TRAINING_STEPS = 100_000
        self.BUFFER_CAPACITY = 50_000
        self.MODEL_TYPE = Q_MODEL.QModel.value


class ConfigAcrobotDuelingDqn(ConfigBase, ConfigAcrobot, ConfigDuelingDqn):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigAcrobot.__init__(self)
        ConfigDuelingDqn.__init__(self)

        self.LEARNING_RATE = 0.001
        self.MAX_TRAINING_STEPS = 100_000
        self.BUFFER_CAPACITY = 50_000
        self.MODEL_TYPE = Q_MODEL.DuelingQModel.value


class ConfigAcrobotDoubleDuelingDqn(ConfigBase, ConfigAcrobot, ConfigDoubleDuelingDqn):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigAcrobot.__init__(self)
        ConfigDoubleDuelingDqn.__init__(self)

        self.LEARNING_RATE = 0.001
        self.MAX_TRAINING_STEPS = 100_000
        self.BUFFER_CAPACITY = 50_000
        self.MODEL_TYPE = Q_MODEL.DuelingQModel.value


# OnPolicy

class ConfigAcrobotReinforce(ConfigBase, ConfigAcrobot, ConfigReinforce):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigAcrobot.__init__(self)
        ConfigReinforce.__init__(self)

        self.MAX_TRAINING_STEPS = 100_000
        self.MODEL_TYPE = VANILLA_POLICY_MODEL.DiscreteVanillaPolicyModel.value


class ConfigAcrobotA2c(ConfigBase, ConfigAcrobot, ConfigA2c):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigAcrobot.__init__(self)
        ConfigA2c.__init__(self)

        self.MAX_TRAINING_STEPS = 100_000
        self.MODEL_TYPE = BASIC_ACTOR_CRITIC_MODEL.DiscreteBasicActorCriticSharedModel.value


class ConfigAcrobotA3c(ConfigBase, ConfigAcrobot, ConfigA3c):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigAcrobot.__init__(self)
        ConfigA3c.__init__(self)

        self.MAX_TRAINING_STEPS = 100_000
        self.MODEL_TYPE = BASIC_ACTOR_CRITIC_MODEL.DiscreteBasicActorCriticSharedModel.value


class ConfigAcrobotPpo(ConfigBase, ConfigAcrobot, ConfigPpo):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigAcrobot.__init__(self)
        ConfigPpo.__init__(self)

        self.MAX_TRAINING_STEPS = 100_000
        self.MODEL_TYPE = BASIC_ACTOR_CRITIC_MODEL.DiscreteBasicActorCriticSharedModel.value


class ConfigAcrobotPpoTrajectory(ConfigBase, ConfigAcrobot, ConfigPpoTrajectory):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigAcrobot.__init__(self)
        ConfigPpoTrajectory.__init__(self)

        self.MAX_TRAINING_STEPS = 100_000
        self.MODEL_TYPE = BASIC_ACTOR_CRITIC_MODEL.DiscreteBasicActorCriticSharedModel.value
