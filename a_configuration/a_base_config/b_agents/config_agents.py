class ConfigAgent:
    def __init__(self):
        self.AGENT_TYPE = None

        self.LEARNING_RATE = None
        self.GAMMA = 0.99
        self.BATCH_SIZE = 256


class ConfigOffPolicyAgent(ConfigAgent):
    def __init__(self):
        super(ConfigOffPolicyAgent, self).__init__()
        self.TEST_INTERVAL_TRAINING_STEPS = 1000

        self.ACTOR_LEARNING_RATE = 0.0002
        self.LEARNING_RATE = 0.0001


class ConfigOnPolicyAgent(ConfigAgent):
    def __init__(self):
        ConfigAgent.__init__(self)
        self.TEST_INTERVAL_TRAINING_STEPS = 200
        self.ENTROPY_BETA = 0.001

        self.USE_GAE = False
        self.GAE_LAMBDA = 0.95
        self.USE_GAE_RECALCULATE_TARGET_VALUE = True

        self.USE_BOOTSTRAP_FOR_TARGET_VALUE = False

        self.ACTOR_LEARNING_RATE = 0.00001
        self.LEARNING_RATE = 0.0001
        self.ENTROPY_BETA = 0.001