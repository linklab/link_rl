class ConfigGymClassicControl:
    pass


class ConfigCartPole(ConfigGymClassicControl):
    def __init__(self):
        self.ENV_NAME = "CartPole-v1"
        self.EPISODE_REWARD_MIN_SOLVED = 450


class ConfigAcrobot(ConfigGymClassicControl):
    def __init__(self):
        self.ENV_NAME = "Acrobot-v1"
        self.EPISODE_REWARD_MIN_SOLVED = -70


class ConfigMountainCar(ConfigGymClassicControl):
    def __init__(self):
        self.ENV_NAME = "MountainCar-v0"
        self.EPISODE_REWARD_MIN_SOLVED = -110
        self.N_TEST_EPISODES = 10
