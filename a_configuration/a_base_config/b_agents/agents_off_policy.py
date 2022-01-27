from torch import nn

from a_configuration.a_base_config.b_agents.agents import ConfigAgent
from g_utils.commons import AgentType


class ConfigDqn(ConfigAgent):
    def __init__(self):
        ConfigAgent.__init__(self)
        self.AGENT_TYPE = AgentType.DQN

        self.LEARNING_RATE = 0.001

        self.EPSILON_INIT = 1.0
        self.EPSILON_FINAL = 0.1
        self.EPSILON_FINAL_TRAINING_STEP_PROPORTION = 0.5

        self.BUFFER_CAPACITY = 10_000
        self.BATCH_SIZE = 64
        self.MIN_BUFFER_SIZE_FOR_TRAIN = self.BATCH_SIZE * 10
        self.TARGET_SYNC_INTERVAL_TRAINING_STEPS = 1_000


class ConfigDoubleDqn(ConfigDqn):
    def __init__(self):
        super(ConfigDoubleDqn, self).__init__()
        self.AGENT_TYPE = AgentType.DOUBLE_DQN

        self.TAU = 0.005
        del self.TARGET_SYNC_INTERVAL_TRAINING_STEPS


class ConfigDuelingDqn(ConfigDqn):
    def __init__(self):
        super(ConfigDuelingDqn, self).__init__()
        self.AGENT_TYPE = AgentType.DUELING_DQN


class ConfigDoubleDuelingDqn(ConfigDqn):
    def __init__(self):
        super(ConfigDoubleDuelingDqn, self).__init__()
        self.AGENT_TYPE = AgentType.DOUBLE_DUELING_DQN

        self.TAU = 0.005
        del self.TARGET_SYNC_INTERVAL_TRAINING_STEPS


class ConfigDdpg(ConfigAgent):
    def __init__(self):
        ConfigAgent.__init__(self)
        self.AGENT_TYPE = AgentType.DDPG

        self.ACTOR_LEARNING_RATE = 0.0001
        self.LEARNING_RATE = 0.001

        self.BUFFER_CAPACITY = 10_000
        self.BATCH_SIZE = 64
        self.MIN_BUFFER_SIZE_FOR_TRAIN = self.BATCH_SIZE * 10
        self.TARGET_SYNC_INTERVAL_TRAINING_STEPS = 50
        self.TAU = 0.005


class ConfigTd3(ConfigAgent):
    def __init__(self):
        ConfigAgent.__init__(self)
        self.AGENT_TYPE = AgentType.TD3

        self.ACTOR_LEARNING_RATE = 0.0001
        self.LEARNING_RATE = 0.001

        self.BUFFER_CAPACITY = 10_000
        self.BATCH_SIZE = 64
        self.MIN_BUFFER_SIZE_FOR_TRAIN = self.BATCH_SIZE * 10
        self.TARGET_SYNC_INTERVAL_TRAINING_STEPS = 50
        self.TAU = 0.005


class ConfigSac(ConfigAgent):
    def __init__(self):
        ConfigAgent.__init__(self)
        self.AGENT_TYPE = AgentType.SAC

        self.LEARNING_RATE = 0.001
        self.ACTOR_LEARNING_RATE = 0.0001
        self.ALPHA_LEARNING_RATE = 0.0003

        self.BUFFER_CAPACITY = 10_000
        self.BATCH_SIZE = 64
        self.MIN_BUFFER_SIZE_FOR_TRAIN = self.BATCH_SIZE * 10
        self.TARGET_SYNC_INTERVAL_TRAINING_STEPS = 50

        self.LAYER_ACTIVATION = nn.ReLU

        self.USE_LAYER_NORM = True

        self.DEFAULT_ALPHA = 0.2
        self.TAU = 0.005
        self.POLICY_UPDATE_FREQUENCY_PER_TRAINING_STEP = 2

        self.AUTOMATIC_ENTROPY_TEMPERATURE_TUNING = True
        self.MIN_ALPHA = 0.2
