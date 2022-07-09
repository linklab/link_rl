from link_rl.a_configuration.a_base_config.b_agents.config_agents_off_policy import ConfigDqn, ConfigDoubleDqn, ConfigDuelingDqn, \
    ConfigDoubleDuelingDqn
from link_rl.a_configuration.a_base_config.b_agents.config_agents_on_policy import ConfigA2c, ConfigReinforce, ConfigPpo, \
    ConfigPpoTrajectory, ConfigA3c
from link_rl.a_configuration.a_base_config.config_single_base import ConfigBase
from link_rl.a_configuration.a_base_config.a_environments.open_ai_gym.config_gym_classic_control import ConfigMountainCar
from link_rl.c_models_v2.b_q_model import Q_MODEL
from link_rl.c_models_v2.c_vanilla_policy_model import VANILLA_POLICY_MODEL
from link_rl.c_models_v2.d_basic_actor_critic_model import BASIC_ACTOR_CRITIC_MODEL


class ConfigMountainCarDqn(ConfigBase, ConfigMountainCar, ConfigDqn):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigMountainCar.__init__(self)
        ConfigDqn.__init__(self)

        self.LEARNING_RATE = 0.001
        self.MAX_TRAINING_STEPS = 100_000
        self.BUFFER_CAPACITY = 50_000
        self.MODEL_TYPE = Q_MODEL.QModel.value


class ConfigMountainCarDoubleDqn(ConfigBase, ConfigMountainCar, ConfigDoubleDqn):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigMountainCar.__init__(self)
        ConfigDoubleDqn.__init__(self)

        self.LEARNING_RATE = 0.001
        self.MAX_TRAINING_STEPS = 100_000
        self.BUFFER_CAPACITY = 50_000
        self.MODEL_TYPE = Q_MODEL.QModel.value


class ConfigMountainCarDuelingDqn(ConfigBase, ConfigMountainCar, ConfigDuelingDqn):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigMountainCar.__init__(self)
        ConfigDuelingDqn.__init__(self)

        self.LEARNING_RATE = 0.001
        self.MAX_TRAINING_STEPS = 100_000
        self.BUFFER_CAPACITY = 50_000
        self.MODEL_TYPE = Q_MODEL.DuelingQModel.value


class ConfigMountainCarDoubleDuelingDqn(ConfigBase, ConfigMountainCar, ConfigDoubleDuelingDqn):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigMountainCar.__init__(self)
        ConfigDoubleDuelingDqn.__init__(self)

        self.LEARNING_RATE = 0.001
        self.MAX_TRAINING_STEPS = 100_000
        self.BUFFER_CAPACITY = 50_000
        self.MODEL_TYPE = Q_MODEL.DuelingQModel.value


# OnPolicy

class ConfigMountainCarReinforce(ConfigBase, ConfigMountainCar, ConfigReinforce):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigMountainCar.__init__(self)
        ConfigReinforce.__init__(self)

        self.MAX_TRAINING_STEPS = 100_000
        self.MODEL_TYPE = VANILLA_POLICY_MODEL.DiscreteVanillaPolicyModel.value


class ConfigMountainCarA2c(ConfigBase, ConfigMountainCar, ConfigA2c):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigMountainCar.__init__(self)
        ConfigA2c.__init__(self)

        self.MAX_TRAINING_STEPS = 100_000
        self.MODEL_TYPE = BASIC_ACTOR_CRITIC_MODEL.DiscreteBasicActorCriticSharedModel.value


class ConfigMountainCarA3c(ConfigBase, ConfigMountainCar, ConfigA3c):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigMountainCar.__init__(self)
        ConfigA3c.__init__(self)

        self.MAX_TRAINING_STEPS = 100_000
        self.MODEL_TYPE = BASIC_ACTOR_CRITIC_MODEL.DiscreteBasicActorCriticSharedModel.value


class ConfigMountainCarPpo(ConfigBase, ConfigMountainCar, ConfigPpo):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigMountainCar.__init__(self)
        ConfigPpo.__init__(self)

        self.MAX_TRAINING_STEPS = 100_000
        self.MODEL_TYPE = BASIC_ACTOR_CRITIC_MODEL.DiscreteBasicActorCriticSharedModel.value


class ConfigMountainCarPpoTrajectory(ConfigBase, ConfigMountainCar, ConfigPpoTrajectory):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigMountainCar.__init__(self)
        ConfigPpoTrajectory.__init__(self)

        self.MAX_TRAINING_STEPS = 100_000
        self.MODEL_TYPE = BASIC_ACTOR_CRITIC_MODEL.DiscreteBasicActorCriticSharedModel.value
