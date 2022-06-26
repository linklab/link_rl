class ConfigGymBox2D:
    pass


class ConfigLunarLander(ConfigGymBox2D):
    def __init__(self):
        self.ENV_NAME = "LunarLander-v2"
        self.EPISODE_REWARD_MIN_SOLVED = 190


class ConfigLunarLanderContinuous(ConfigGymBox2D):
    def __init__(self):
        self.ENV_NAME = "LunarLanderContinuous-v2"
        self.EPISODE_REWARD_MIN_SOLVED = 190


class ConfigNormalBipedalWalker(ConfigGymBox2D):
    def __init__(self):
        self.ENV_NAME = "NormalBipedalWalker-v3"
        self.EPISODE_REWARD_MIN_SOLVED = 300


class ConfigHardcoreBipedalWalker(ConfigGymBox2D):
    def __init__(self):
        self.ENV_NAME = "HardcoreBipedalWalker-v3"
        self.EPISODE_REWARD_MIN_SOLVED = 300


class ConfigCarRacing(ConfigGymBox2D):
    def __init__(self):
        self.ENV_NAME = "CarRacing-v1"
        self.EPISODE_REWARD_MIN_SOLVED = 900
