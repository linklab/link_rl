from a_configuration.a_base_config.a_environments.pybullet.gym_pybullet import ConfigHopperBullet
from a_configuration.a_base_config.b_agents.agents_off_policy import ConfigSac
from a_configuration.a_base_config.b_agents.agents_on_policy import ConfigPpoTrajectory
from a_configuration.a_base_config.config_single_base import ConfigBase
from g_utils.commons import print_basic_info, get_env_info
from g_utils.types import ModelType


class ConfigHopperBulletSac(ConfigBase, ConfigHopperBullet, ConfigSac):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigHopperBullet.__init__(self)
        ConfigSac.__init__(self)

        self.BUFFER_CAPACITY = 250_000
        self.MAX_TRAINING_STEPS = 2_000_000
        self.MODEL_TYPE = ModelType.SMALL_LINEAR


class ConfigHopperBulletPpoTrajectory(ConfigBase, ConfigHopperBullet, ConfigPpoTrajectory):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigHopperBullet.__init__(self)
        ConfigPpoTrajectory.__init__(self)

        self.MAX_TRAINING_STEPS = 2_000_000
        self.MODEL_TYPE = ModelType.SMALL_LINEAR


if __name__ == "__main__":
    config = ConfigHopperBulletSac()
    observation_space, action_space = get_env_info(config)
    print_basic_info(observation_space=observation_space, action_space=action_space, config=config)
