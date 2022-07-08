from link_rl.a_configuration.a_base_config.a_environments.open_ai_gym.config_gym_mujoco import \
    ConfigInvertedDoublePendulumMujoco
from link_rl.a_configuration.a_base_config.b_agents.config_agents_on_policy import ConfigPpo
from link_rl.a_configuration.a_base_config.config_single_base import ConfigBase
from link_rl.c_models_v2.d_basic_actor_critic_model import BASIC_ACTOR_CRITIC_MODEL


class ConfigInvertedDoublePendulumMujocoPpo(ConfigBase, ConfigInvertedDoublePendulumMujoco, ConfigPpo):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigInvertedDoublePendulumMujoco.__init__(self)
        ConfigPpo.__init__(self)

        self.MAX_TRAINING_STEPS = 2_000_000
        self.MODEL_TYPE = BASIC_ACTOR_CRITIC_MODEL.DiscreteBasicActorCriticSharedModel.value


