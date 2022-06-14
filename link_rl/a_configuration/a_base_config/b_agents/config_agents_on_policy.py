from link_rl.a_configuration.a_base_config.b_agents.config_agents import ConfigOnPolicyAgent
from link_rl.g_utils.commons import AgentType


class ConfigReinforce(ConfigOnPolicyAgent):
    def __init__(self):
        ConfigOnPolicyAgent.__init__(self)
        self.AGENT_TYPE = AgentType.REINFORCE


class ConfigA2c(ConfigOnPolicyAgent):
    def __init__(self):
        ConfigOnPolicyAgent.__init__(self)
        self.AGENT_TYPE = AgentType.A2C

        self.USE_GAE = False


class ConfigA3c(ConfigOnPolicyAgent):
    def __init__(self):
        ConfigOnPolicyAgent.__init__(self)
        self.AGENT_TYPE = AgentType.A3C

        self.N_ACTORS = 2
        self.USE_GAE = False


class ConfigPpo(ConfigOnPolicyAgent):
    def __init__(self):
        ConfigOnPolicyAgent.__init__(self)
        self.AGENT_TYPE = AgentType.PPO

        self.USE_GAE = True
        self.USE_GAE_RECALCULATE_TARGET_VALUE = True

        self.PPO_EPSILON_CLIP = 0.2
        self.PPO_K_EPOCH = 3


class ConfigAsynchronousPpo(ConfigOnPolicyAgent):
    def __init__(self):
        ConfigOnPolicyAgent.__init__(self)
        self.AGENT_TYPE = AgentType.ASYNCHRONOUS_PPO

        self.USE_GAE = True
        self.USE_GAE_RECALCULATE_TARGET_VALUE = True

        self.PPO_EPSILON_CLIP = 0.2
        self.PPO_K_EPOCH = 3
        self.N_ACTORS = 2


class ConfigPpoTrajectory(ConfigOnPolicyAgent):
    def __init__(self):
        ConfigOnPolicyAgent.__init__(self)
        self.AGENT_TYPE = AgentType.PPO_TRAJECTORY

        self.USE_GAE = True
        self.USE_GAE_RECALCULATE_TARGET_VALUE = True

        self.PPO_EPSILON_CLIP = 0.2
        self.PPO_K_EPOCH = 3
