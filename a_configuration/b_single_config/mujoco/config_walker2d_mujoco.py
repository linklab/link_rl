from a_configuration.a_base_config.a_environments.pybullet.gym_mujoco import ConfigWalker2dMujoco
from a_configuration.a_base_config.b_agents.agents_off_policy import ConfigSac
from a_configuration.a_base_config.b_agents.agents_on_policy import ConfigPpoTrajectory
from a_configuration.a_base_config.config_single_base import ConfigBase
from g_utils.types import ModelType


class ConfigWalker2dMujocoSac(ConfigBase, ConfigWalker2dMujoco, ConfigSac):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigWalker2dMujoco.__init__(self)
        ConfigSac.__init__(self)

        self.BUFFER_CAPACITY = 1_000_000
        self.MAX_TRAINING_STEPS = 2_000_000
        self.MODEL_TYPE = ModelType.SMALL_LINEAR


class ConfigWalker2dMujocoPpoTrajectory(ConfigBase, ConfigWalker2dMujoco, ConfigPpoTrajectory):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigWalker2dMujoco.__init__(self)
        ConfigPpoTrajectory.__init__(self)

        self.MAX_TRAINING_STEPS = 2_000_000
        self.MODEL_TYPE = ModelType.SMALL_LINEAR
