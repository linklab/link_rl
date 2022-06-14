from link_rl.a_configuration.a_base_config.a_environments.open_ai_gym.config_gym_mujoco import ConfigWalker2dMujoco
from link_rl.a_configuration.a_base_config.b_agents.config_agents_off_policy import ConfigSac
from link_rl.a_configuration.a_base_config.b_agents.config_agents_on_policy import ConfigPpoTrajectory, ConfigPpo
from link_rl.a_configuration.a_base_config.config_single_base import ConfigBase
from link_rl.g_utils.types import ModelType


class ConfigWalker2dMujocoSac(ConfigBase, ConfigWalker2dMujoco, ConfigSac):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigWalker2dMujoco.__init__(self)
        ConfigSac.__init__(self)

        self.BUFFER_CAPACITY = 1_000_000
        self.MAX_TRAINING_STEPS = 2_000_000
        self.MODEL_TYPE = ModelType.MEDIUM_LINEAR


class ConfigWalker2dMujocoPpo(ConfigBase, ConfigWalker2dMujoco, ConfigPpo):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigWalker2dMujoco.__init__(self)
        ConfigPpo.__init__(self)

        self.MAX_TRAINING_STEPS = 2_000_000
        self.MODEL_TYPE = ModelType.MEDIUM_LINEAR


class ConfigWalker2dMujocoPpoTrajectory(ConfigBase, ConfigWalker2dMujoco, ConfigPpoTrajectory):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigWalker2dMujoco.__init__(self)
        ConfigPpoTrajectory.__init__(self)

        self.MAX_TRAINING_STEPS = 2_000_000
        self.MODEL_TYPE = ModelType.MEDIUM_LINEAR
