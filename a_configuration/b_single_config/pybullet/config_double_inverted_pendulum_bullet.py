from a_configuration.a_base_config.a_environments.pybullet.gym_pybullet import ConfigAntBullet, \
    ConfigDoubleInvertedPendulumBullet
from a_configuration.a_base_config.b_agents.agents_off_policy import ConfigDqn, ConfigDdpg, ConfigSac
from a_configuration.a_base_config.b_agents.agents_on_policy import ConfigA2c, ConfigReinforce
from a_configuration.a_base_config.c_models.linear_models import ConfigLinearModel
from a_configuration.a_base_config.config_single_base import ConfigBase
from g_utils.commons import print_basic_info, get_env_info
from g_utils.types import ModelType


class ConfigDoubleInvertedPendulumBulletA2c(ConfigBase, ConfigDoubleInvertedPendulumBullet, ConfigA2c):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigDoubleInvertedPendulumBullet.__init__(self)
        ConfigA2c.__init__(self)

        self.BATCH_SIZE = 64
        self.N_STEP = 1
        self.BUFFER_CAPACITY = self.BATCH_SIZE
        self.MIN_BUFFER_SIZE_FOR_TRAIN = self.BATCH_SIZE * 10

        self.N_VECTORIZED_ENVS = 1
        self.N_ACTORS = 1
        self.MAX_TRAINING_STEPS = 2_000_000
        self.MODEL_TYPE = ModelType.SMALL_LINEAR


class ConfigDoubleInvertedPendulumBulletDdpg(ConfigBase, ConfigDoubleInvertedPendulumBullet, ConfigDdpg):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigDoubleInvertedPendulumBullet.__init__(self)
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


class ConfigDoubleInvertedPendulumBulletSac(ConfigBase, ConfigDoubleInvertedPendulumBullet, ConfigSac):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigDoubleInvertedPendulumBullet.__init__(self)
        ConfigSac.__init__(self)

        self.ALPHA_LEARNING_RATE = 0.0001
        self.ACTOR_LEARNING_RATE = 0.0002
        self.LEARNING_RATE = 0.001

        self.BATCH_SIZE = 64

        self.N_STEP = 1
        self.BUFFER_CAPACITY = 250_000
        self.MIN_BUFFER_SIZE_FOR_TRAIN = self.BATCH_SIZE * 10

        self.N_VECTORIZED_ENVS = 1
        self.N_ACTORS = 1
        self.MAX_TRAINING_STEPS = 2_000_000
        self.MODEL_TYPE = ModelType.SMALL_LINEAR


if __name__ == "__main__":
    config = ConfigDoubleInvertedPendulumBulletSac()
    observation_space, action_space = get_env_info(config)
    print_basic_info(observation_space=observation_space, action_space=action_space, config=config)