from link_rl.a_configuration.a_base_config.a_environments.open_ai_gym.config_gym_mujoco import ConfigHopperMujoco
from link_rl.a_configuration.a_base_config.b_agents.config_agents_off_policy import ConfigSac
from link_rl.a_configuration.a_base_config.b_agents.config_agents_on_policy import ConfigPpoTrajectory, ConfigPpo
from link_rl.a_configuration.a_base_config.config_single_base import ConfigBase
from link_rl.g_utils.types import ModelType


class ConfigHopperMujocoSac(ConfigBase, ConfigHopperMujoco, ConfigSac):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigHopperMujoco.__init__(self)
        ConfigSac.__init__(self)

        self.BUFFER_CAPACITY = 1_000_000
        self.MAX_TRAINING_STEPS = 2_000_000
        self.MODEL_TYPE = ModelType.MEDIUM_LINEAR


class ConfigHopperMujocoPpo(ConfigBase, ConfigHopperMujoco, ConfigPpo):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigHopperMujoco.__init__(self)
        ConfigPpo.__init__(self)

        self.MAX_TRAINING_STEPS = 2_000_000
        self.MODEL_TYPE = ModelType.MEDIUM_LINEAR


class ConfigHopperMujocoPpoTrajectory(ConfigBase, ConfigHopperMujoco, ConfigPpoTrajectory):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigHopperMujoco.__init__(self)
        ConfigPpoTrajectory.__init__(self)

        self.MAX_TRAINING_STEPS = 2_000_000
        self.MODEL_TYPE = ModelType.MEDIUM_LINEAR
