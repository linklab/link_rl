class ParameterAntMujoco:
    def __init__(self):
        self.ENV_NAME = "Ant-v2"
        self.EPISODE_REWARD_AVG_SOLVED = 2000
        self.EPISODE_REWARD_STD_SOLVED = 100
        self.TEST_INTERVAL_TRAINING_STEPS = 1_024