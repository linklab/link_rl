from a_configuration.a_base_config.b_agents.config_agents import ConfigOnPolicyAgent
from g_utils.commons import AgentType


class ConfigReinforce(ConfigOnPolicyAgent):
    def __init__(self):
        ConfigOnPolicyAgent.__init__(self)
        self.AGENT_TYPE = AgentType.REINFORCE


class ConfigA2c(ConfigOnPolicyAgent):
    def __init__(self):
        ConfigOnPolicyAgent.__init__(self)
        self.AGENT_TYPE = AgentType.A2C


class ConfigA3c(ConfigOnPolicyAgent):
    def __init__(self):
        ConfigOnPolicyAgent.__init__(self)
        self.AGENT_TYPE = AgentType.A3C
        self.N_ACTORS = 2


class ConfigPpo(ConfigOnPolicyAgent):
    def __init__(self):
        ConfigOnPolicyAgent.__init__(self)
        self.AGENT_TYPE = AgentType.PPO

        self.PPO_EPSILON_CLIP = 0.2
        self.PPO_K_EPOCH = 3


class ConfigPpoTrajectory(ConfigOnPolicyAgent):
    def __init__(self):
        ConfigOnPolicyAgent.__init__(self)
        self.AGENT_TYPE = AgentType.PPO_TRAJECTORY

        self.PPO_EPSILON_CLIP = 0.2
        self.PPO_K_EPOCH = 3
