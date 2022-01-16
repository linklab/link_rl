class ParameterGymBox:
    pass

class Parameter3DBall(ParameterGymBox):
    def __init__(self):
        self.ENV_NAME = "Unity3DBall"
        self.EPISODE_REWARD_AVG_SOLVED = 100
        self.EPISODE_REWARD_STD_SOLVED = 10
        self.TEST_INTERVAL_TRAINING_STEPS = 1024