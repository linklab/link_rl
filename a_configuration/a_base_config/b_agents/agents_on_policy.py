from a_configuration.a_base_config.b_agents.agents import ConfigOnPolicyAgent
from g_utils.commons import AgentType


class ConfigReinforce(ConfigOnPolicyAgent):
    def __init__(self):
        ConfigOnPolicyAgent.__init__(self)
        self.AGENT_TYPE = AgentType.REINFORCE

        self.LEARNING_RATE = 0.0001


class ConfigA2c(ConfigOnPolicyAgent):
    def __init__(self):
        ConfigOnPolicyAgent.__init__(self)
        self.AGENT_TYPE = AgentType.A2C

        self.ACTOR_LEARNING_RATE = 0.0001
        self.LEARNING_RATE = 0.001

        self.ENTROPY_BETA = 0.001

        self.BATCH_SIZE = 128


class ConfigPpo(ConfigOnPolicyAgent):
    def __init__(self):
        ConfigOnPolicyAgent.__init__(self)
        self.AGENT_TYPE = AgentType.PPO

        self.ACTOR_LEARNING_RATE = 0.00001
        self.LEARNING_RATE = 0.00005

        self.ENTROPY_BETA = 0.001

        self.PPO_EPSILON_CLIP = 0.2
        self.BATCH_SIZE = 128
        self.PPO_K_EPOCH = 3


class ConfigPpoTrajectory(ConfigOnPolicyAgent):
    def __init__(self):
        ConfigOnPolicyAgent.__init__(self)
        self.AGENT_TYPE = AgentType.PPO_TRAJECTORY

        self.ACTOR_LEARNING_RATE = 0.00001
        self.LEARNING_RATE = 0.00005

        self.ENTROPY_BETA = 0.001

        self.PPO_EPSILON_CLIP = 0.2
        self.BATCH_SIZE = 128
        self.PPO_K_EPOCH = 3
