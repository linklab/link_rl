from torch import nn

from a_configuration.a_base_config.a_environments.unity.unity_box import ConfigWalker
from a_configuration.a_base_config.b_agents.agents_off_policy import ConfigDdpg, ConfigSac
from a_configuration.a_base_config.config_single_base import ConfigBase
from g_utils.types import ModelType


class ConfigWalkerDdqg(ConfigBase, ConfigWalker, ConfigDdpg):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigWalker.__init__(self)
        ConfigDdpg.__init__(self)

        self.BATCH_SIZE = 64
        self.ACTOR_LEARNING_RATE = 0.0002
        self.LEARNING_RATE = 0.001
        self.N_STEP = 1
        self.BUFFER_CAPACITY = 250_000
        self.MIN_BUFFER_SIZE_FOR_TRAIN = self.BATCH_SIZE * 10

        self.N_VECTORIZED_ENVS = 1
        self.N_ACTORS = 1
        self.MAX_TRAINING_STEPS = 2_000_000
        self.MODEL_TYPE = ModelType.SMALL_LINEAR

class ConfigWalkerSac(ConfigBase, ConfigWalker, ConfigSac):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigWalker.__init__(self)
        ConfigSac.__init__(self)

        self.BUFFER_CAPACITY = 2_000_000
        self.MAX_TRAINING_STEPS = 15_000_000
        self.MODEL_TYPE = ModelType.SMALL_LINEAR