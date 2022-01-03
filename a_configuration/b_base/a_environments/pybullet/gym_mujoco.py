class ParameterAntMujoco:
    def __init__(self):
        self.ENV_NAME = "Ant-v2"
        self.EPISODE_REWARD_AVG_SOLVED = 5000
        self.EPISODE_REWARD_STD_SOLVED = 300
        self.TEST_INTERVAL_TRAINING_STEPS = 1_024