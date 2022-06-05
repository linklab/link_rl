from a_configuration.a_base_config.b_agents.config_agents import ConfigOffPolicyAgent
from g_utils.commons import AgentType


class ConfigDqn(ConfigOffPolicyAgent):
    def __init__(self):
        super(ConfigDqn, self).__init__()
        self.AGENT_TYPE = AgentType.DQN

        self.EPSILON_INIT = 1.0
        self.EPSILON_FINAL = 0.1
        self.EPSILON_FINAL_TRAINING_STEP_PROPORTION = 0.5

        self.BUFFER_CAPACITY = 10_000
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

        self.TAU = 0.005
        self.BUFFER_CAPACITY = 10_000


class ConfigTd3(ConfigOffPolicyAgent):
    def __init__(self):
        super(ConfigTd3, self).__init__()
        self.AGENT_TYPE = AgentType.TD3

        self.TAU = 0.005
        self.BUFFER_CAPACITY = 10_000

        self.POLICY_UPDATE_FREQUENCY_PER_TRAINING_STEP = 2


class ConfigSac(ConfigOffPolicyAgent):
    def __init__(self):
        super(ConfigSac, self).__init__()
        self.AGENT_TYPE = AgentType.SAC

        self.TAU = 0.005
        self.BUFFER_CAPACITY = 2_000_000
        self.TARGET_SYNC_INTERVAL_TRAINING_STEPS = 50

        self.POLICY_UPDATE_FREQUENCY_PER_TRAINING_STEP = 2

        self.DEFAULT_ALPHA = 1.0
        self.AUTOMATIC_ENTROPY_TEMPERATURE_TUNING = True
        self.ALPHA_LEARNING_RATE = 0.0002
        self.MIN_ALPHA = 0.01


class ConfigMuzero(ConfigOffPolicyAgent):
    def __init__(self):
        super(ConfigMuzero, self).__init__()
        self.AGENT_TYPE = AgentType.MUZERO

        self.LEARNING_RATE = 0.001
        self.BUFFER_CAPACITY = 10_000  # Episode-based

        self.TEST_INTERVAL_TRAINING_STEPS = 300
        self.MAX_TRAINING_STEPS = 10000
        self.STACKED_OBSERVATION = 0
        self.INDEX_STACKED_OBSERVATIONS = -1

        self.ROOT_DIRCHLET_ALPHA = 0.25
        self.ROOT_EXPLORATION_FRACTION = 0.25

        self.PB_C_BASE = 19652
        self.PB_C_INIT = 1.25

        self.NUM_SIMULATION = 50
        self.PLAYERS = list(range(1))

        self.SUPPORT_SIZE = 10

        self.NUM_UNROLL_STEPS = 5

        self.VALUE_LOSS_WEIGHT = 0.25
