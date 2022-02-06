class ConfigAgent:
    def __init__(self):
        self.AGENT_TYPE = None

        self.LEARNING_RATE = None
        self.GAMMA = 0.99


class ConfigOffPolicyAgent(ConfigAgent):
    def __init__(self):
        super(ConfigOffPolicyAgent, self).__init__()
        self.TEST_INTERVAL_TRAINING_STEPS = 1000


class ConfigOnPolicyAgent(ConfigAgent):
    def __init__(self):
        ConfigAgent.__init__(self)
        self.TEST_INTERVAL_TRAINING_STEPS = 300


class ConfigMuzero(ConfigAgent):
    def __init__(self):
        ConfigAgent.__init__(self)
        self.TEST_INTERVAL_TRAINING_STEPS = 300
        self.MAX_TRAINING_STEPS = 10000
        self.STACKED_OBSERVATION = 0
        self.INDEX_STACKED_OBSERVATIONS = -1
