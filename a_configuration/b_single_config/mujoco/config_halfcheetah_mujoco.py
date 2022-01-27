from a_configuration.a_base_config.a_environments.pybullet.gym_mujoco import ConfigHalfCheetahMujoco
from a_configuration.a_base_config.b_agents.agents_off_policy import ConfigSac
from a_configuration.a_base_config.config_single_base import ConfigBase
from g_utils.types import ModelType
from torch import nn


class ConfigHalfCheetahMujocoSac(ConfigBase, ConfigHalfCheetahMujoco, ConfigSac):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigHalfCheetahMujoco.__init__(self)
        ConfigSac.__init__(self)

        self.BATCH_SIZE = 256

        self.ALPHA_LEARNING_RATE = 0.0001
        self.ACTOR_LEARNING_RATE = 0.0002
        self.LEARNING_RATE = 0.001
        self.N_STEP = 2
        self.BUFFER_CAPACITY = 1_000_000
        self.MIN_BUFFER_SIZE_FOR_TRAIN = self.BATCH_SIZE * 10

        self.N_VECTORIZED_ENVS = 1
        self.N_ACTORS = 1
        self.MAX_TRAINING_STEPS = 2_000_000

        self.MODEL_TYPE = ModelType.SMALL_LINEAR

        self.USE_LAYER_NORM = False
