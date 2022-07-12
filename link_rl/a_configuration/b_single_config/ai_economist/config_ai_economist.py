from link_rl.a_configuration.a_base_config.a_environments.ai_economist.config_ai_economist import ConfigAiEconomist
from link_rl.a_configuration.a_base_config.b_agents.config_agents import ConfigAiEconomistAgent
from link_rl.a_configuration.a_base_config.config_single_base import ConfigBase
from link_rl.d_models.d_basic_actor_critic_model import BASIC_ACTOR_CRITIC_MODEL
from link_rl.h_utils.types import AgentType


class ConfigAiEconomistPpo(ConfigBase, ConfigAiEconomist, ConfigAiEconomistAgent):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigAiEconomist.__init__(self)
        ConfigAiEconomistAgent.__init__(self)

        self.MAX_TRAINING_STEPS = 100_000
        self.MODEL_TYPE = BASIC_ACTOR_CRITIC_MODEL.DiscreteBasicActorCriticSharedModel.value
        self.INTERNAL_AGENT_TYPE = AgentType.PPO

