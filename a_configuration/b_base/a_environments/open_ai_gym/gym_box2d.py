class ParameterGymBox2D:
    pass


class ParameterLunarLander(ParameterGymBox2D):
    def __init__(self):
        self.ENV_NAME = "LunarLander-v2"
        self.EPISODE_REWARD_AVG_SOLVED = 190
        self.EPISODE_REWARD_STD_SOLVED = 25.0
        self.TEST_INTERVAL_TRAINING_STEPS = 1024


class ParameterLunarLanderContinuous(ParameterGymBox2D):
    def __init__(self):
        self.ENV_NAME = "LunarLanderContinuous-v2"
        self.EPISODE_REWARD_AVG_SOLVED = 190
        self.EPISODE_REWARD_STD_SOLVED = 25.0
        self.TEST_INTERVAL_TRAINING_STEPS = 1024
