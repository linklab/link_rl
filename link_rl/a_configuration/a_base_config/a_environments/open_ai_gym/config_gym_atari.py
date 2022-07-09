from link_rl.a_configuration.a_base_config.a_environments.open_ai_gym import ConfigGymAtari


class ConfigPong(ConfigGymAtari):
    def __init__(self):
        super(ConfigPong, self).__init__()
        self.ENV_NAME = "ALE/Pong-v5"
        self.EPISODE_REWARD_MIN_SOLVED = 20


class ConfigBreakout(ConfigGymAtari):
    def __init__(self):
        super(ConfigBreakout, self).__init__()
        self.ENV_NAME = "ALE/Breakout-v5"
        self.EPISODE_REWARD_MIN_SOLVED = 700


class ConfigVideoPinball(ConfigGymAtari):
    def __init__(self):
        super(ConfigVideoPinball, self).__init__()
        self.ENV_NAME = "ALE/VideoPinball-v5"
        self.EPISODE_REWARD_MIN_SOLVED = 900_000


class ConfigDemonAttack(ConfigGymAtari):
    def __init__(self):
        super(ConfigDemonAttack, self).__init__()
        self.ENV_NAME = "ALE/DemonAttack-v5"
        self.EPISODE_REWARD_MIN_SOLVED = 900_000
