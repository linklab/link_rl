from a_configuration.a_base_config.b_agents.agents import ConfigOffPolicyAgent
from g_utils.commons import AgentType


class ConfigDqn(ConfigOffPolicyAgent):
    def __init__(self):
        super(ConfigDqn, self).__init__()
        self.AGENT_TYPE = AgentType.DQN

        self.LEARNING_RATE = 0.001

        self.EPSILON_INIT = 1.0
        self.EPSILON_FINAL = 0.1
        self.EPSILON_FINAL_TRAINING_STEP_PROPORTION = 0.5

        self.BUFFER_CAPACITY = 10_000
        self.BATCH_SIZE = 128
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


class ConfigDdpg(ConfigOffPolicyAgent):
    def __init__(self):
        super(ConfigDdpg, self).__init__()
        self.AGENT_TYPE = AgentType.DDPG

        self.ACTOR_LEARNING_RATE = 0.0001
        self.LEARNING_RATE = 0.001

        self.TAU = 0.005
        self.BUFFER_CAPACITY = 10_000
        self.BATCH_SIZE = 128

class ConfigTd3(ConfigOffPolicyAgent):
    def __init__(self):
        super(ConfigTd3, self).__init__()
        self.AGENT_TYPE = AgentType.TD3

        self.ACTOR_LEARNING_RATE = 0.0001
        self.LEARNING_RATE = 0.001

        self.TAU = 0.005
        self.BUFFER_CAPACITY = 10_000
        self.BATCH_SIZE = 128

        self.POLICY_UPDATE_FREQUENCY_PER_TRAINING_STEP = 2


class ConfigSac(ConfigOffPolicyAgent):
    def __init__(self):
        super(ConfigSac, self).__init__()
        self.AGENT_TYPE = AgentType.SAC

        self.ACTOR_LEARNING_RATE = 0.0002
        self.LEARNING_RATE = 0.001
        self.ALPHA_LEARNING_RATE = 0.00001

        self.TAU = 0.005
        self.BUFFER_CAPACITY = 10_000
        self.BATCH_SIZE = 128
        self.TARGET_SYNC_INTERVAL_TRAINING_STEPS = 50

        self.POLICY_UPDATE_FREQUENCY_PER_TRAINING_STEP = 2

        self.DEFAULT_ALPHA = 1.0
        self.AUTOMATIC_ENTROPY_TEMPERATURE_TUNING = True
        self.MIN_ALPHA = 0.2


class ConfigMuzero(ConfigOffPolicyAgent):
    def __init__(self):
        super(ConfigMuzero, self).__init__()
        self.AGENT_TYPE = AgentType.MUZERO

        self.LEARNING_RATE = 0.001
        self.BUFFER_CAPACITY = 10_000  # Episode-based
        self.BATCH_SIZE = 128

        self.TEST_INTERVAL_TRAINING_STEPS = 300
        self.MAX_TRAINING_STEPS = 10000
        self.STACKED_OBSERVATION = 0
        self.INDEX_STACKED_OBSERVATIONS = -1

        self.SUPPORT_SIZE = 10