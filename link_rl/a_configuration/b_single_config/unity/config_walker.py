from link_rl.a_configuration.a_base_config.a_environments.unity.config_unity_box import ConfigWalker
from link_rl.a_configuration.a_base_config.b_agents.config_agents_off_policy import ConfigDdpg, ConfigSac
from link_rl.a_configuration.a_base_config.config_single_base import ConfigBase
from link_rl.g_utils.types import ModelType


class ConfigWalkerDdqg(ConfigBase, ConfigWalker, ConfigDdpg):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigWalker.__init__(self)
        ConfigDdpg.__init__(self)

        self.N_STEP = 1
        self.BUFFER_CAPACITY = 250_000

        self.MAX_TRAINING_STEPS = 2_000_000
        self.MODEL_TYPE = ModelType.MEDIUM_LINEAR


class ConfigWalkerSac(ConfigBase, ConfigWalker, ConfigSac):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigWalker.__init__(self)
        ConfigSac.__init__(self)

        self.BUFFER_CAPACITY = 2_000_000
        self.MAX_TRAINING_STEPS = 15_000_000
        self.MODEL_TYPE = ModelType.SMALL_LINEAR
