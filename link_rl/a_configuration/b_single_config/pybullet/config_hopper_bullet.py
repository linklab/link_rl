from link_rl.a_configuration.a_base_config.a_environments.pybullet.config_gym_pybullet import ConfigHopperBullet
from link_rl.a_configuration.a_base_config.b_agents.config_agents_off_policy import ConfigSac
from link_rl.a_configuration.a_base_config.b_agents.config_agents_on_policy import ConfigPpoTrajectory, ConfigPpo
from link_rl.a_configuration.a_base_config.config_single_base import ConfigBase
from link_rl.c_models_v2.d_basic_actor_critic_model import BASIC_ACTOR_CRITIC_MODEL
from link_rl.c_models_v2.g_sac_model import SAC_MODEL
from link_rl.g_utils.commons import print_basic_info, get_env_info


class ConfigHopperBulletSac(ConfigBase, ConfigHopperBullet, ConfigSac):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigHopperBullet.__init__(self)
        ConfigSac.__init__(self)

        self.BUFFER_CAPACITY = 250_000
        self.MAX_TRAINING_STEPS = 2_000_000
        self.MODEL_TYPE = SAC_MODEL.ContinuousSacEncoderModel.value


class ConfigHopperBulletPpo(ConfigBase, ConfigHopperBullet, ConfigPpo):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigHopperBullet.__init__(self)
        ConfigPpo.__init__(self)

        self.MAX_TRAINING_STEPS = 2_000_000
        self.MODEL_TYPE = BASIC_ACTOR_CRITIC_MODEL.ContinuousBasicActorCriticEncoderSharedModel.value


class ConfigHopperBulletPpoTrajectory(ConfigBase, ConfigHopperBullet, ConfigPpoTrajectory):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigHopperBullet.__init__(self)
        ConfigPpoTrajectory.__init__(self)

        self.MAX_TRAINING_STEPS = 2_000_000
        self.MODEL_TYPE = BASIC_ACTOR_CRITIC_MODEL.ContinuousBasicActorCriticEncoderSharedModel.value


if __name__ == "__main__":
    config = ConfigHopperBulletSac()
    observation_space, action_space = get_env_info(config)
    print_basic_info(observation_space=observation_space, action_space=action_space, config=config)
