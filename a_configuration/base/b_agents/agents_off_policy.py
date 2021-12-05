from g_utils.commons import AgentType


class ParameterDqn:
    def __init__(self):
        self.AGENT_TYPE = AgentType.Dqn

        self.LEARNING_RATE = 0.0001

        self.EPSILON_INIT = 1.0
        self.EPSILON_FINAL = 0.1
        self.EPSILON_FINAL_TIME_STEP_PERCENT = 0.35

        self.BUFFER_CAPACITY = 10_000
        self.BATCH_SIZE = 64
        self.MIN_BUFFER_SIZE_FOR_TRAIN = self.BATCH_SIZE
        self.GAMMA = 0.99
        self.TARGET_SYNC_INTERVAL_TRAINING_STEPS = 50
