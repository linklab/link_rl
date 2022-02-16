from a_configuration.a_base_config.a_environments.mujoco.config_gym_mujoco import ConfigHalfCheetahMujoco
from a_configuration.a_base_config.b_agents.config_agents_off_policy import ConfigSac
from a_configuration.a_base_config.b_agents.config_agents_on_policy import ConfigPpoTrajectory, ConfigPpo
from a_configuration.a_base_config.config_single_base import ConfigBase
from g_utils.types import ModelType


class ConfigHalfCheetahMujocoSac(ConfigBase, ConfigHalfCheetahMujoco, ConfigSac):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigHalfCheetahMujoco.__init__(self)
        ConfigSac.__init__(self)

        self.BUFFER_CAPACITY = 1_000_000
        self.MAX_TRAINING_STEPS = 2_000_000
        self.MODEL_TYPE = ModelType.MEDIUM_LINEAR


class ConfigHalfCheetahMujocoPpo(ConfigBase, ConfigHalfCheetahMujoco, ConfigPpo):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigHalfCheetahMujoco.__init__(self)
        ConfigPpo.__init__(self)

        self.MAX_TRAINING_STEPS = 2_000_000
        self.MODEL_TYPE = ModelType.MEDIUM_LINEAR


class ConfigHalfCheetahMujocoPpoTrajectory(ConfigBase, ConfigHalfCheetahMujoco, ConfigPpoTrajectory):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigHalfCheetahMujoco.__init__(self)
        ConfigPpoTrajectory.__init__(self)

        self.MAX_TRAINING_STEPS = 2_000_000
        self.MODEL_TYPE = ModelType.MEDIUM_LINEAR