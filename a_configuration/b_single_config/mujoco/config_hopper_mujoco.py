from a_configuration.a_base_config.a_environments.mujoco.gym_mujoco import ConfigHopperMujoco
from a_configuration.a_base_config.b_agents.agents_off_policy import ConfigSac
from a_configuration.a_base_config.b_agents.agents_on_policy import ConfigPpoTrajectory
from a_configuration.a_base_config.config_single_base import ConfigBase
from g_utils.types import ModelType


class ConfigHopperMujocoSac(ConfigBase, ConfigHopperMujoco, ConfigSac):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigHopperMujoco.__init__(self)
        ConfigSac.__init__(self)

        self.BUFFER_CAPACITY = 1_000_000
        self.MAX_TRAINING_STEPS = 2_000_000
        self.MODEL_TYPE = ModelType.SMALL_LINEAR


class ConfigHopperMujocoPpoTrajectory(ConfigBase, ConfigHopperMujoco, ConfigPpoTrajectory):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigHopperMujoco.__init__(self)
        ConfigPpoTrajectory.__init__(self)

        self.MAX_TRAINING_STEPS = 2_000_000
        self.MODEL_TYPE = ModelType.SMALL_LINEAR
