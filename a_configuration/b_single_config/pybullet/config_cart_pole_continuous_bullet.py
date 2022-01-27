from a_configuration.a_base_config.a_environments.pybullet.gym_pybullet import ConfigCartPoleContinuousBullet
from a_configuration.a_base_config.b_agents.agents_off_policy import ConfigSac, ConfigDdpg, ConfigTd3
from a_configuration.a_base_config.b_agents.agents_on_policy import ConfigA2c, ConfigPpo
from a_configuration.a_base_config.config_single_base import ConfigBase
from g_utils.types import ModelType


class ConfigCartPoleContinuousBulletA2c(
    ConfigBase, ConfigCartPoleContinuousBullet, ConfigA2c
):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigCartPoleContinuousBullet.__init__(self)
        ConfigA2c.__init__(self)

        self.N_VECTORIZED_ENVS = 1
        self.N_ACTORS = 1
        self.BATCH_SIZE = 512
        self.MAX_TRAINING_STEPS = 100_000
        self.MODEL_TYPE = ModelType.SMALL_LINEAR
        self.ACTOR_LEARNING_RATE = 0.0005
        self.LEARNING_RATE = 0.001


class ConfigCartPoleContinuousBulletPpo(
    ConfigBase, ConfigCartPoleContinuousBullet, ConfigPpo
):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigCartPoleContinuousBullet.__init__(self)
        ConfigPpo.__init__(self)

        self.N_VECTORIZED_ENVS = 1
        self.N_ACTORS = 1
        self.MAX_TRAINING_STEPS = 100_000
        self.MODEL_TYPE = ModelType.SMALL_LINEAR

        self.BATCH_SIZE = 256
        self.PPO_TRAJECTORY_SIZE = self.BATCH_SIZE * 10
        self.BUFFER_CAPACITY = self.PPO_TRAJECTORY_SIZE
        self.ACTOR_LEARNING_RATE = 0.0005
        self.LEARNING_RATE = 0.001


class ConfigCartPoleContinuousBulletSac(
    ConfigBase, ConfigCartPoleContinuousBullet, ConfigSac
):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigCartPoleContinuousBullet.__init__(self)
        ConfigSac.__init__(self)

        self.ALPHA_LEARNING_RATE = 0.0001
        self.LEARNING_RATE = 0.001
        self.N_VECTORIZED_ENVS = 1
        self.N_ACTORS = 1
        self.MAX_TRAINING_STEPS = 100_000
        self.MODEL_TYPE = ModelType.SMALL_LINEAR
        self.ACTOR_LEARNING_RATE = 0.0005
        self.LEARNING_RATE = 0.001


class ConfigCartPoleContinuousBulletDdpg(
    ConfigBase, ConfigCartPoleContinuousBullet, ConfigDdpg
):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigCartPoleContinuousBullet.__init__(self)
        ConfigDdpg.__init__(self)

        self.LEARNING_RATE = 0.001
        self.N_VECTORIZED_ENVS = 1
        self.BUFFER_CAPACITY = 200_000
        self.N_ACTORS = 1
        self.MAX_TRAINING_STEPS = 200_000
        self.MODEL_TYPE = ModelType.SMALL_LINEAR
        # self.MODEL_TYPE = ModelType.SMALL_RECURRENT


class ConfigCartPoleContinuousBulletTd3(
    ConfigBase, ConfigCartPoleContinuousBullet, ConfigTd3
):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigCartPoleContinuousBullet.__init__(self)
        ConfigTd3.__init__(self)

        self.LEARNING_RATE = 0.001
        self.N_VECTORIZED_ENVS = 1
        self.BUFFER_CAPACITY = 200_000
        self.N_ACTORS = 1
        self.MAX_TRAINING_STEPS = 200_000
        self.MODEL_TYPE = ModelType.SMALL_LINEAR
        # self.MODEL_TYPE = ModelType.SMALL_RECURRENT