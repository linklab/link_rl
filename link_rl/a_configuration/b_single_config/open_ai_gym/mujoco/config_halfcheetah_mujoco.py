from link_rl.a_configuration.a_base_config.a_environments.open_ai_gym.config_gym_mujoco import ConfigHalfCheetahMujoco
from link_rl.a_configuration.a_base_config.b_agents.config_agents_off_policy import ConfigSac
from link_rl.a_configuration.a_base_config.b_agents.config_agents_on_policy import ConfigPpoTrajectory, ConfigPpo
from link_rl.a_configuration.a_base_config.config_single_base import ConfigBase
from link_rl.d_models.d_basic_actor_critic_model import BASIC_ACTOR_CRITIC_MODEL
from link_rl.d_models.g_sac_model import SAC_MODEL


class ConfigHalfCheetahMujocoSac(ConfigBase, ConfigHalfCheetahMujoco, ConfigSac):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigHalfCheetahMujoco.__init__(self)
        ConfigSac.__init__(self)

        self.BUFFER_CAPACITY = 1_000_000
        self.MAX_TRAINING_STEPS = 2_000_000
        self.MODEL_TYPE = SAC_MODEL.ContinuousSacModel.value


class ConfigHalfCheetahMujocoPpo(ConfigBase, ConfigHalfCheetahMujoco, ConfigPpo):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigHalfCheetahMujoco.__init__(self)
        ConfigPpo.__init__(self)

        self.MAX_TRAINING_STEPS = 2_000_000
        self.MODEL_TYPE = BASIC_ACTOR_CRITIC_MODEL.ContinuousBasicActorCriticSharedModel.value


class ConfigHalfCheetahMujocoPpoTrajectory(ConfigBase, ConfigHalfCheetahMujoco, ConfigPpoTrajectory):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigHalfCheetahMujoco.__init__(self)
        ConfigPpoTrajectory.__init__(self)

        self.MAX_TRAINING_STEPS = 2_000_000
        self.MODEL_TYPE = BASIC_ACTOR_CRITIC_MODEL.ContinuousBasicActorCriticSharedModel.value