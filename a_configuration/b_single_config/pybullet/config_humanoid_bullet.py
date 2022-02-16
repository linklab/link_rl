from a_configuration.a_base_config.a_environments.pybullet.config_gym_pybullet import ConfigHumanoidBullet
from a_configuration.a_base_config.b_agents.config_agents_off_policy import ConfigSac
from a_configuration.a_base_config.b_agents.config_agents_on_policy import ConfigPpo, ConfigPpoTrajectory
from a_configuration.a_base_config.config_single_base import ConfigBase
from g_utils.commons import print_basic_info, get_env_info
from g_utils.types import ModelType


class ConfigHumanoidBulletSac(ConfigBase, ConfigHumanoidBullet, ConfigSac):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigHumanoidBullet.__init__(self)
        ConfigSac.__init__(self)

        self.BUFFER_CAPACITY = 250_000
        self.MAX_TRAINING_STEPS = 2_000_000
        self.MODEL_TYPE = ModelType.MEDIUM_LINEAR


class ConfigHumanoidBulletPpo(ConfigBase, ConfigHumanoidBullet, ConfigPpo):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigHumanoidBullet.__init__(self)
        ConfigPpo.__init__(self)

        self.MAX_TRAINING_STEPS = 2_000_000
        self.MODEL_TYPE = ModelType.MEDIUM_LINEAR


class ConfigHumanoidBulletPpoTrajectory(ConfigBase, ConfigHumanoidBullet, ConfigPpoTrajectory):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigHumanoidBullet.__init__(self)
        ConfigPpoTrajectory.__init__(self)

        self.MAX_TRAINING_STEPS = 2_000_000
        self.MODEL_TYPE = ModelType.MEDIUM_LINEAR
