class ParameterGymBox2D:
    pass


class ParameterLunarLander(ParameterGymBox2D):
    def __init__(self):
        self.ENV_NAME = "LunarLander-v2"
        self.EPISODE_REWARD_AVG_SOLVED = 190
        self.EPISODE_REWARD_STD_SOLVED = 25.0


class ParameterLunarLanderContinuous(ParameterGymBox2D):
    def __init__(self):
        self.ENV_NAME = "LunarLanderContinuous-v2"
        self.EPISODE_REWARD_AVG_SOLVED = 190
        self.EPISODE_REWARD_STD_SOLVED = 25.0
