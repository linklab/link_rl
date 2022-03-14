class ConfigGymAtari:
    pass


class ConfigPong(ConfigGymAtari):
    def __init__(self):
        self.ENV_NAME = "ALE/Pong-v5"
        self.EPISODE_REWARD_AVG_SOLVED = 20
        self.EPISODE_REWARD_STD_SOLVED = 1.0


class ConfigBreakout(ConfigGymAtari):
    def __init__(self):
        self.ENV_NAME = "ALE/Breakout-v5"
        self.EPISODE_REWARD_AVG_SOLVED = 700
        self.EPISODE_REWARD_STD_SOLVED = 20.0


class ConfigVideoPinball(ConfigGymAtari):
    def __init__(self):
        self.ENV_NAME = "ALE/VideoPinball-v5"
        self.EPISODE_REWARD_AVG_SOLVED = 900_000
        self.EPISODE_REWARD_STD_SOLVED = 100.0
