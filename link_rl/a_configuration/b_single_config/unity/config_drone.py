from link_rl.a_configuration.a_base_config.a_environments.unity.config_unity_box import ConfigDrone
from link_rl.a_configuration.a_base_config.b_agents.config_agents_off_policy import ConfigDdpg, ConfigSac
from link_rl.a_configuration.a_base_config.config_single_base import ConfigBase
from link_rl.g_utils.types import ModelType


class ConfigDroneDdpg(ConfigBase, ConfigDrone, ConfigDdpg):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigDrone.__init__(self)
        ConfigDdpg.__init__(self)

        self.BATCH_SIZE = 256
        self.N_STEP = 1
        self.BUFFER_CAPACITY = 250_000

        self.MAX_TRAINING_STEPS = 2_000_000
        self.MODEL_TYPE = ModelType.SMALL_2D_CONVOLUTIONAL

class ConfigDroneSac(ConfigBase, ConfigDrone, ConfigSac):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigDrone.__init__(self)
        ConfigSac.__init__(self)

        self.BUFFER_CAPACITY = 1_000_000
        self.MAX_TRAINING_STEPS = 30_000_000
        self.MODEL_TYPE = ModelType.SMALL_LINEAR

        self.ACTOR_LEARNING_RATE = 0.0005
        self.POLICY_UPDATE_FREQUENCY_PER_TRAINING_STEP = 128



