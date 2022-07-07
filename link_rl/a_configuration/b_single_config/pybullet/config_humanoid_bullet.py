from link_rl.a_configuration.a_base_config.a_environments.pybullet.config_gym_pybullet import ConfigHumanoidBullet
from link_rl.a_configuration.a_base_config.b_agents.config_agents_off_policy import ConfigSac
from link_rl.a_configuration.a_base_config.b_agents.config_agents_on_policy import ConfigPpo, ConfigPpoTrajectory
from link_rl.a_configuration.a_base_config.config_single_base import ConfigBase
from link_rl.c_models_v2.d_basic_actor_critic_model import BASIC_ACTOR_CRITIC_MODEL
from link_rl.c_models_v2.g_sac_model import SAC_MODEL
from link_rl.g_utils.types import ModelType


class ConfigHumanoidBulletSac(ConfigBase, ConfigHumanoidBullet, ConfigSac):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigHumanoidBullet.__init__(self)
        ConfigSac.__init__(self)

        self.BUFFER_CAPACITY = 250_000
        self.MAX_TRAINING_STEPS = 2_000_000
        self.MODEL_TYPE = SAC_MODEL.ContinuousSacModel.value


class ConfigHumanoidBulletPpo(ConfigBase, ConfigHumanoidBullet, ConfigPpo):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigHumanoidBullet.__init__(self)
        ConfigPpo.__init__(self)

        self.MAX_TRAINING_STEPS = 2_000_000
        self.MODEL_TYPE = BASIC_ACTOR_CRITIC_MODEL.ContinuousBasicActorCriticSharedModel.value


class ConfigHumanoidBulletPpoTrajectory(ConfigBase, ConfigHumanoidBullet, ConfigPpoTrajectory):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigHumanoidBullet.__init__(self)
        ConfigPpoTrajectory.__init__(self)

        self.MAX_TRAINING_STEPS = 2_000_000
        self.MODEL_TYPE = BASIC_ACTOR_CRITIC_MODEL.ContinuousBasicActorCriticSharedModel.value
