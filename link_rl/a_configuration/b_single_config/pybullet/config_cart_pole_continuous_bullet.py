from link_rl.a_configuration.a_base_config.a_environments.pybullet.config_gym_pybullet import ConfigCartPoleContinuousBullet
from link_rl.a_configuration.a_base_config.b_agents.config_agents_off_policy import ConfigSac, ConfigDdpg, ConfigTd3
from link_rl.a_configuration.a_base_config.b_agents.config_agents_on_policy import ConfigA2c, ConfigPpo, ConfigPpoTrajectory, \
    ConfigReinforce, ConfigA3c
from link_rl.a_configuration.a_base_config.config_single_base import ConfigBase
from link_rl.g_utils.types import ModelType


class ConfigCartPoleContinuousBulletReinforce(ConfigBase, ConfigCartPoleContinuousBullet, ConfigReinforce):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigCartPoleContinuousBullet.__init__(self)
        ConfigReinforce.__init__(self)

        self.MAX_TRAINING_STEPS = 100_000
        self.MODEL_TYPE = ModelType.SMALL_LINEAR


class ConfigCartPoleContinuousBulletA2c(ConfigBase, ConfigCartPoleContinuousBullet, ConfigA2c):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigCartPoleContinuousBullet.__init__(self)
        ConfigA2c.__init__(self)

        self.MAX_TRAINING_STEPS = 100_000
        self.MODEL_TYPE = ModelType.SMALL_LINEAR


class ConfigCartPoleContinuousBulletA3c(ConfigBase, ConfigCartPoleContinuousBullet, ConfigA3c):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigCartPoleContinuousBullet.__init__(self)
        ConfigA3c.__init__(self)

        self.MAX_TRAINING_STEPS = 100_000
        self.MODEL_TYPE = ModelType.SMALL_LINEAR


class ConfigCartPoleContinuousBulletPpo(ConfigBase, ConfigCartPoleContinuousBullet, ConfigPpo):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigCartPoleContinuousBullet.__init__(self)
        ConfigPpo.__init__(self)

        self.MAX_TRAINING_STEPS = 100_000
        self.MODEL_TYPE = ModelType.SMALL_LINEAR


class ConfigCartPoleContinuousBulletPpoTrajectory(ConfigBase, ConfigCartPoleContinuousBullet, ConfigPpoTrajectory):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigCartPoleContinuousBullet.__init__(self)
        ConfigPpoTrajectory.__init__(self)

        self.MAX_TRAINING_STEPS = 100_000
        self.MODEL_TYPE = ModelType.SMALL_LINEAR


class ConfigCartPoleContinuousBulletSac(ConfigBase, ConfigCartPoleContinuousBullet, ConfigSac):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigCartPoleContinuousBullet.__init__(self)
        ConfigSac.__init__(self)

        self.MAX_TRAINING_STEPS = 100_000
        self.MODEL_TYPE = ModelType.SMALL_LINEAR


class ConfigCartPoleContinuousBulletDdpg(ConfigBase, ConfigCartPoleContinuousBullet, ConfigDdpg):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigCartPoleContinuousBullet.__init__(self)
        ConfigDdpg.__init__(self)

        self.MAX_TRAINING_STEPS = 200_000
        self.MODEL_TYPE = ModelType.SMALL_LINEAR
        # self.MODEL_TYPE = ModelType.SMALL_RECURRENT


class ConfigCartPoleContinuousBulletTd3(ConfigBase, ConfigCartPoleContinuousBullet, ConfigTd3):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigCartPoleContinuousBullet.__init__(self)
        ConfigTd3.__init__(self)

        self.MAX_TRAINING_STEPS = 200_000
        self.MODEL_TYPE = ModelType.SMALL_LINEAR
        # self.MODEL_TYPE = ModelType.SMALL_RECURRENT
