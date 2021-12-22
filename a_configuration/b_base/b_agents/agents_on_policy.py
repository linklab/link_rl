from a_configuration.b_base.b_agents.agents import ParameterAgent
from g_utils.commons import AgentType


class ParameterReinforce(ParameterAgent):
    def __init__(self):
        ParameterAgent.__init__(self)
        self.AGENT_TYPE = AgentType.Reinforce

        self.LEARNING_RATE = 0.0001
        self.BUFFER_CAPACITY = 1_000
        self.GAMMA = 0.99


class ParameterA2c(ParameterAgent):
    def __init__(self):
        ParameterAgent.__init__(self)
        self.AGENT_TYPE = AgentType.A2c

        self.LEARNING_RATE = 0.0001
        self.BUFFER_CAPACITY = 1_000
        self.GAMMA = 0.99
        self.BATCH_SIZE = 64
        self.MIN_BUFFER_SIZE_FOR_TRAIN = self.BATCH_SIZE

