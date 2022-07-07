from link_rl.a_configuration.a_base_config.a_environments.pybullet.config_gym_pybullet import ConfigAntBullet
from link_rl.a_configuration.a_base_config.b_agents.config_agents_off_policy import ConfigDdpg, ConfigSac, ConfigTd3
from link_rl.a_configuration.a_base_config.b_agents.config_agents_on_policy import ConfigA2c, ConfigPpo, ConfigPpoTrajectory
from link_rl.a_configuration.a_base_config.config_single_base import ConfigBase
from link_rl.c_models_v2.d_basic_actor_critic_model import BASIC_ACTOR_CRITIC_MODEL
from link_rl.c_models_v2.e_ddpg_model import DDPG_MODEL
from link_rl.c_models_v2.f_td3_model import TD3_MODEL
from link_rl.c_models_v2.g_sac_model import SAC_MODEL
from link_rl.g_utils.commons import print_basic_info, get_env_info
from link_rl.g_utils.types import ModelType


class ConfigAntBulletA2c(ConfigBase, ConfigAntBullet, ConfigA2c):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigAntBullet.__init__(self)
        ConfigA2c.__init__(self)

        self.MAX_TRAINING_STEPS = 2_000_000
        self.MODEL_TYPE = BASIC_ACTOR_CRITIC_MODEL.ContinuousBasicActorCriticSharedModel.value


class ConfigAntBulletPpo(ConfigBase, ConfigAntBullet, ConfigPpo):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigAntBullet.__init__(self)
        ConfigPpo.__init__(self)

        self.MAX_TRAINING_STEPS = 2_000_000
        self.MODEL_TYPE = BASIC_ACTOR_CRITIC_MODEL.ContinuousBasicActorCriticSharedModel.value


class ConfigAntBulletPpoTrajectory(ConfigBase, ConfigAntBullet, ConfigPpoTrajectory):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigAntBullet.__init__(self)
        ConfigPpoTrajectory.__init__(self)

        self.MAX_TRAINING_STEPS = 2_000_000
        self.MODEL_TYPE = BASIC_ACTOR_CRITIC_MODEL.ContinuousBasicActorCriticSharedModel.value


class ConfigAntBulletDdpg(ConfigBase, ConfigAntBullet, ConfigDdpg):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigAntBullet.__init__(self)
        ConfigDdpg.__init__(self)

        self.BUFFER_CAPACITY = 250_000
        self.MAX_TRAINING_STEPS = 2_000_000
        self.MODEL_TYPE = DDPG_MODEL.ContinuousDdpgModel.value


class ConfigAntBulletTd3(ConfigBase, ConfigAntBullet, ConfigTd3):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigAntBullet.__init__(self)
        ConfigTd3.__init__(self)

        self.BUFFER_CAPACITY = 250_000
        self.MAX_TRAINING_STEPS = 2_000_000
        self.MODEL_TYPE = TD3_MODEL.ContinuousTd3Model.value


class ConfigAntBulletSac(ConfigBase, ConfigAntBullet, ConfigSac):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigAntBullet.__init__(self)
        ConfigSac.__init__(self)

        self.BUFFER_CAPACITY = 250_000
        self.MAX_TRAINING_STEPS = 2_000_000
        self.MODEL_TYPE = SAC_MODEL.ContinuousSacModel.value


if __name__ == "__main__":
    config = ConfigAntBulletSac()
    observation_space, action_space = get_env_info(config)
    print_basic_info(observation_space=observation_space, action_space=action_space, config=config)
