from torch import nn

from a_configuration.b_base.b_agents.agents import ParameterAgent
from g_utils.commons import AgentType


class ParameterDqn(ParameterAgent):
    def __init__(self):
        ParameterAgent.__init__(self)
        self.AGENT_TYPE = AgentType.DQN

        self.LEARNING_RATE = 0.0001

        self.EPSILON_INIT = 1.0
        self.EPSILON_FINAL = 0.1
        self.EPSILON_FINAL_TRAINING_STEP_PERCENT = 0.35

        self.BUFFER_CAPACITY = 10_000
        self.BATCH_SIZE = 64
        self.MIN_BUFFER_SIZE_FOR_TRAIN = self.BATCH_SIZE
        self.GAMMA = 0.99
        self.TARGET_SYNC_INTERVAL_TRAINING_STEPS = 50


class ParameterDdqn(ParameterDqn):
    def __init__(self):
        super(ParameterDdqn, self).__init__()
        self.AGENT_TYPE = AgentType.DOUBLE_DQN

        self.TAU = 0.005


class ParameterDdpg(ParameterAgent):
    def __init__(self):
        ParameterAgent.__init__(self)
        self.AGENT_TYPE = AgentType.DDPG

        self.LEARNING_RATE = 0.0001

        self.BUFFER_CAPACITY = 10_000
        self.BATCH_SIZE = 64
        self.MIN_BUFFER_SIZE_FOR_TRAIN = self.BATCH_SIZE
        self.GAMMA = 0.99
        self.TARGET_SYNC_INTERVAL_TRAINING_STEPS = 50
        self.TAU = 0.005


class ParameterSac(ParameterAgent):
    def __init__(self):
        ParameterAgent.__init__(self)
        self.AGENT_TYPE = AgentType.SAC

        self.LEARNING_RATE = 0.001
        self.ACTOR_LEARNING_RATE = 0.0001

        self.BUFFER_CAPACITY = 10_000
        self.BATCH_SIZE = 128
        self.MIN_BUFFER_SIZE_FOR_TRAIN = 200_000
        self.GAMMA = 0.99
        self.TARGET_SYNC_INTERVAL_TRAINING_STEPS = 50

        self.LAYER_ACTIVATION = nn.GELU()

        self.LAYER_NORM = True

        self.ALPHA = 0.2

        self.TAU = 0.005
