from link_rl.a_configuration.a_base_config.a_environments.open_ai_gym import ConfigGymBox2D


class ConfigLunarLander(ConfigGymBox2D):
    def __init__(self):
        super(ConfigLunarLander, self).__init__()
        self.ENV_NAME = "LunarLander-v2"
        self.EPISODE_REWARD_MIN_SOLVED = 190


class ConfigLunarLanderContinuous(ConfigGymBox2D):
    def __init__(self):
        super(ConfigLunarLanderContinuous, self).__init__()
        self.ENV_NAME = "LunarLanderContinuous-v2"
        self.EPISODE_REWARD_MIN_SOLVED = 190


class ConfigNormalBipedalWalker(ConfigGymBox2D):
    def __init__(self):
        super(ConfigNormalBipedalWalker, self).__init__()
        self.ENV_NAME = "NormalBipedalWalker-v3"
        self.EPISODE_REWARD_MIN_SOLVED = 300


class ConfigHardcoreBipedalWalker(ConfigGymBox2D):
    def __init__(self):
        super(ConfigHardcoreBipedalWalker, self).__init__()
        self.ENV_NAME = "HardcoreBipedalWalker-v3"
        self.EPISODE_REWARD_MIN_SOLVED = 300


class ConfigCarRacing(ConfigGymBox2D):
    def __init__(self):
        super(ConfigCarRacing, self).__init__()
        self.ENV_NAME = "CarRacing-v1"
        self.EPISODE_REWARD_MIN_SOLVED = 900
