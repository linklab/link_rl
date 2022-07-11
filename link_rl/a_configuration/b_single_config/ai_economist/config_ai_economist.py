from link_rl.a_configuration.a_base_config.a_environments.ai_economist.config_ai_economist import ConfigAiEconomist
from link_rl.a_configuration.a_base_config.b_agents.config_agents_on_policy import ConfigPpo
from link_rl.a_configuration.a_base_config.config_single_base import ConfigBase
from link_rl.c_models_v2.d_basic_actor_critic_model import BASIC_ACTOR_CRITIC_MODEL


class ConfigAiEconomistPpo(ConfigBase, ConfigAiEconomist, ConfigPpo):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigAiEconomist.__init__(self)
        ConfigPpo.__init__(self)

        self.MAX_TRAINING_STEPS = 100_000
        self.MODEL_TYPE = BASIC_ACTOR_CRITIC_MODEL.DiscreteBasicActorCriticSharedModel

        self.CONSOLE_LOG_INTERVAL_TRAINING_STEPS = self.PPO_K_EPOCH
