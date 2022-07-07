from link_rl.a_configuration.a_base_config.a_environments.pybullet.config_gym_pybullet import ConfigCartPoleBullet
from link_rl.a_configuration.a_base_config.b_agents.config_agents_off_policy import ConfigDqn, \
    ConfigDoubleDqn, ConfigDuelingDqn, ConfigDoubleDuelingDqn
from link_rl.a_configuration.a_base_config.b_agents.config_agents_on_policy import ConfigA2c, ConfigPpo, ConfigPpoTrajectory, \
    ConfigA3c, ConfigReinforce
from link_rl.a_configuration.a_base_config.config_single_base import ConfigBase
from link_rl.c_models_v2.b_q_model import Q_MODEL
from link_rl.c_models_v2.c_vanilla_policy_model import VANILLA_POLICY_MODEL
from link_rl.c_models_v2.d_basic_actor_critic_model import BASIC_ACTOR_CRITIC_MODEL
from link_rl.g_utils.types import ModelType


class ConfigCartPoleBulletDqn(ConfigBase, ConfigCartPoleBullet, ConfigDqn):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigCartPoleBullet.__init__(self)
        ConfigDqn.__init__(self)

        self.MAX_TRAINING_STEPS = 100_000
        self.MODEL_TYPE = Q_MODEL.QModel.value


class ConfigCartPoleBulletDoubleDqn(ConfigBase, ConfigCartPoleBullet, ConfigDoubleDqn):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigCartPoleBullet.__init__(self)
        ConfigDoubleDqn.__init__(self)

        self.MAX_TRAINING_STEPS = 100_000
        self.MODEL_TYPE = Q_MODEL.QModel.value


class ConfigCartPoleBulletDuelingDqn(ConfigBase, ConfigCartPoleBullet, ConfigDuelingDqn):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigCartPoleBullet.__init__(self)
        ConfigDuelingDqn.__init__(self)

        self.MAX_TRAINING_STEPS = 100_000
        self.MODEL_TYPE = Q_MODEL.DuelingQModel.value


class ConfigCartPoleBulletDoubleDuelingDqn(ConfigBase, ConfigCartPoleBullet, ConfigDoubleDuelingDqn):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigCartPoleBullet.__init__(self)
        ConfigDoubleDuelingDqn.__init__(self)

        self.MAX_TRAINING_STEPS = 100_000
        self.MODEL_TYPE = Q_MODEL.DuelingQModel.value


class ConfigCartPoleBulletReinforce(ConfigBase, ConfigCartPoleBullet, ConfigReinforce):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigCartPoleBullet.__init__(self)
        ConfigReinforce.__init__(self)

        self.MAX_TRAINING_STEPS = 100_000
        self.MODEL_TYPE = VANILLA_POLICY_MODEL.DiscreteVanillaPolicyModel.value


class ConfigCartPoleBulletA2c(ConfigBase, ConfigCartPoleBullet, ConfigA2c):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigCartPoleBullet.__init__(self)
        ConfigA2c.__init__(self)

        self.MAX_TRAINING_STEPS = 100_000
        self.MODEL_TYPE = BASIC_ACTOR_CRITIC_MODEL.DiscreteBasicActorCriticSharedModel.value


class ConfigCartPoleBulletA3c(ConfigBase, ConfigCartPoleBullet, ConfigA3c):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigCartPoleBullet.__init__(self)
        ConfigA3c.__init__(self)

        self.MAX_TRAINING_STEPS = 100_000
        self.MODEL_TYPE = BASIC_ACTOR_CRITIC_MODEL.DiscreteBasicActorCriticSharedModel.value


class ConfigCartPoleBulletPpo(ConfigBase, ConfigCartPoleBullet, ConfigPpo):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigCartPoleBullet.__init__(self)
        ConfigPpo.__init__(self)

        self.MAX_TRAINING_STEPS = 100_000
        self.MODEL_TYPE = BASIC_ACTOR_CRITIC_MODEL.DiscreteBasicActorCriticSharedModel.value


class ConfigCartPoleBulletPpoTrajectory(ConfigBase, ConfigCartPoleBullet, ConfigPpoTrajectory):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigCartPoleBullet.__init__(self)
        ConfigPpoTrajectory.__init__(self)

        self.MAX_TRAINING_STEPS = 100_000
        self.MODEL_TYPE = BASIC_ACTOR_CRITIC_MODEL.DiscreteBasicActorCriticSharedModel.value
