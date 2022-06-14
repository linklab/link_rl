from link_rl.a_configuration.a_base_config.a_environments.gym_robotics.config_gym_robotics_hand_manipulate_block_rotate_xyz import \
    ConfigHandManipulateBlockRotateXYZ
from link_rl.a_configuration.a_base_config.b_agents.config_agents_off_policy import ConfigTd3, ConfigSac
from link_rl.a_configuration.a_base_config.b_agents.config_agents_on_policy import ConfigA2c, ConfigPpo
from link_rl.a_configuration.a_base_config.config_single_base import ConfigBase
from link_rl.g_utils.types import ModelType


class ConfigHandManipulateBlockRotateXYZA2c(ConfigBase, ConfigHandManipulateBlockRotateXYZ, ConfigA2c):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigHandManipulateBlockRotateXYZA2c.__init__(self)
        ConfigA2c.__init__(self)

        self.MAX_TRAINING_STEPS = 1_000_000
        self.MODEL_TYPE = ModelType.MEDIUM_LINEAR


class ConfigHandManipulateBlockRotateXYZPpo(ConfigBase, ConfigHandManipulateBlockRotateXYZ, ConfigPpo):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigHandManipulateBlockRotateXYZ.__init__(self)
        ConfigPpo.__init__(self)

        self.MAX_TRAINING_STEPS = 1_000_000
        self.MODEL_TYPE = ModelType.MEDIUM_LINEAR


class ConfigHandManipulateBlockRotateXYZTd3(ConfigBase, ConfigHandManipulateBlockRotateXYZ, ConfigTd3):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigHandManipulateBlockRotateXYZ.__init__(self)
        ConfigTd3.__init__(self)

        self.MAX_TRAINING_STEPS = 1_000_000
        self.MODEL_TYPE = ModelType.MEDIUM_LINEAR


class ConfigHandManipulateBlockRotateXYZSac(ConfigBase, ConfigHandManipulateBlockRotateXYZ, ConfigSac):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigHandManipulateBlockRotateXYZ.__init__(self)
        ConfigSac.__init__(self)

        self.MAX_TRAINING_STEPS = 1_000_000
        self.MODEL_TYPE = ModelType.MEDIUM_LINEAR
