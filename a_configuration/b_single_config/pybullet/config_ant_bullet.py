from a_configuration.a_base_config.a_environments.pybullet.gym_pybullet import ConfigAntBullet
from a_configuration.a_base_config.b_agents.agents_off_policy import ConfigDdpg, ConfigSac, ConfigTd3
from a_configuration.a_base_config.b_agents.agents_on_policy import ConfigA2c, ConfigPpo, ConfigPpoTrajectory
from a_configuration.a_base_config.config_single_base import ConfigBase
from g_utils.commons import print_basic_info, get_env_info
from g_utils.types import ModelType


class ConfigAntBulletA2c(ConfigBase, ConfigAntBullet, ConfigA2c):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigAntBullet.__init__(self)
        ConfigA2c.__init__(self)

        self.MAX_TRAINING_STEPS = 2_000_000
        self.MODEL_TYPE = ModelType.MEDIUM_LINEAR


class ConfigAntBulletPpo(ConfigBase, ConfigAntBullet, ConfigPpo):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigAntBullet.__init__(self)
        ConfigPpo.__init__(self)

        self.MAX_TRAINING_STEPS = 2_000_000
        self.MODEL_TYPE = ModelType.MEDIUM_LINEAR


class ConfigAntBulletPpoTrajectory(ConfigBase, ConfigAntBullet, ConfigPpoTrajectory):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigAntBullet.__init__(self)
        ConfigPpoTrajectory.__init__(self)

        self.MAX_TRAINING_STEPS = 2_000_000
        self.MODEL_TYPE = ModelType.MEDIUM_LINEAR


class ConfigAntBulletDdpg(ConfigBase, ConfigAntBullet, ConfigDdpg):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigAntBullet.__init__(self)
        ConfigDdpg.__init__(self)

        self.BUFFER_CAPACITY = 250_000
        self.MAX_TRAINING_STEPS = 2_000_000
        self.MODEL_TYPE = ModelType.MEDIUM_LINEAR


class ConfigAntBulletTd3(ConfigBase, ConfigAntBullet, ConfigTd3):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigAntBullet.__init__(self)
        ConfigTd3.__init__(self)

        self.BUFFER_CAPACITY = 250_000
        self.MAX_TRAINING_STEPS = 2_000_000
        self.MODEL_TYPE = ModelType.MEDIUM_LINEAR


class ConfigAntBulletSac(ConfigBase, ConfigAntBullet, ConfigSac):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigAntBullet.__init__(self)
        ConfigSac.__init__(self)

        self.BUFFER_CAPACITY = 250_000
        self.MAX_TRAINING_STEPS = 2_000_000
        self.MODEL_TYPE = ModelType.SMALL_LINEAR


if __name__ == "__main__":
    config = ConfigAntBulletSac()
    observation_space, action_space = get_env_info(config)
    print_basic_info(observation_space=observation_space, action_space=action_space, config=config)
