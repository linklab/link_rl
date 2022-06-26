class ConfigGymAtari:
    pass


class ConfigPong(ConfigGymAtari):
    def __init__(self):
        self.ENV_NAME = "ALE/Pong-v5"
        self.EPISODE_REWARD_MIN_SOLVED = 20
        self.FRAME_SKIP = 4


class ConfigBreakout(ConfigGymAtari):
    def __init__(self):
        self.ENV_NAME = "ALE/Breakout-v5"
        self.EPISODE_REWARD_MIN_SOLVED = 700
        self.FRAME_SKIP = 4


class ConfigVideoPinball(ConfigGymAtari):
    def __init__(self):
        self.ENV_NAME = "ALE/VideoPinball-v5"
        self.EPISODE_REWARD_MIN_SOLVED = 900_000
        self.FRAME_SKIP = 4
