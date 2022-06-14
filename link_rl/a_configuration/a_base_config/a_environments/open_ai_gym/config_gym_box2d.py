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


class ConfigNormalBipedalWalker(ConfigGymBox2D):
    def __init__(self):
        self.ENV_NAME = "NormalBipedalWalker-v3"
        self.EPISODE_REWARD_AVG_SOLVED = 300
        self.EPISODE_REWARD_STD_SOLVED = 20.0


class ConfigHardcoreBipedalWalker(ConfigGymBox2D):
    def __init__(self):
        self.ENV_NAME = "HardcoreBipedalWalker-v3"
        self.EPISODE_REWARD_AVG_SOLVED = 300
        self.EPISODE_REWARD_STD_SOLVED = 20.0


class ConfigCarRacing(ConfigGymBox2D):
    def __init__(self):
        self.ENV_NAME = "CarRacing-v1"
        self.EPISODE_REWARD_AVG_SOLVED = 900
        self.EPISODE_REWARD_STD_SOLVED = 10.0