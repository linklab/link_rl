from link_rl.a_configuration.a_base_config.b_agents.config_agents import ConfigOffPolicyAgent
from link_rl.g_utils.commons import AgentType


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


class ConfigTdmpc(ConfigOffPolicyAgent):
    def __init__(self):
        super(ConfigTdmpc, self).__init__()
        self.AGENT_TYPE = AgentType.TDMPC

        self.TAU = 0.01
        self.BUFFER_CAPACITY = 1_000_000
        self.TARGET_SYNC_INTERVAL_TRAINING_STEPS = 50

        self.ITERATIONS = 6
        self.NUM_SAMPLES = 512
        self.NUM_ELITES = 64
        self.MIXTURE_COEF = 0.05
        self.MIN_STD = 0.05
        self.TEMPERATURE = 0.5
        self.MOMENTUM = 0.1


        self.BATCH_SIZE = 512
        self.BUFFER_CAPACITY = 1000000
        self.HORIZON = 5
        self.REWARD_COEF = 0.5
        self.VALUE_COEF = 0.1
        self.CONSISTENCY_COEF = 2
        self.RHO = 0.5
        self.LEARNING_RATE = 0.001

        self.STD_SCHEDULE = 'linear(0.5, {}, 25000)'.format(self.MIN_STD)
        self.HORIZON_SCHEDULE = 'linear(1, {}, 25000)'.format(self.HORIZON)

        self.MAX_TRAINING_STEPS = 1_000_000
        self.POLICY_UPDATE_FREQUENCY_PER_TRAINING_STEP = 2
        self.TAU = 0.01
        self.CLIP_GRADIENT_VALUE = 10

        self.TRAIN_INTERVAL_GLOBAL_TIME_STEPS = 1

        self.N_STEP = 1

        self.USE_PER = False

        self.IMG_SIZE = 84
        self.ACTION_REPEAT = 4
        self.FRAME_STACK = 3
        self.SEED_STEPS = 5000

        self.LATENT_DIM = 50
        self.ENC_DIM = 256
        self.MLP_DIM = 512
        self.NUM_CHANNELS = 32

        self.TARGET_MODEL_UPDATE_FREQ = 2
        self.TEST_INTERVAL_TRAINING_STEPS = 2500