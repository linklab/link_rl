class ConfigGymBox2D:
    pass


class ConfigLunarLander(ConfigGymBox2D):
    def __init__(self):
        self.ENV_NAME = "LunarLander-v2"
        self.EPISODE_REWARD_AVG_SOLVED = 190
        self.EPISODE_REWARD_STD_SOLVED = 25.0


class ConfigLunarLanderContinuous(ConfigGymBox2D):
    def __init__(self):
        self.ENV_NAME = "LunarLanderContinuous-v2"
        self.EPISODE_REWARD_AVG_SOLVED = 190
        self.EPISODE_REWARD_STD_SOLVED = 25.0
