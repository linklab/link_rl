class ConfigGymClassicControl:
    pass


class ConfigCartPole(ConfigGymClassicControl):
    def __init__(self):
        self.ENV_NAME = "CartPole-v1"
        self.EPISODE_REWARD_AVG_SOLVED = 450
        self.EPISODE_REWARD_STD_SOLVED = 50.0


class ConfigAcrobot(ConfigGymClassicControl):
    def __init__(self):
        self.ENV_NAME = "Acrobot-v1"
        self.EPISODE_REWARD_AVG_SOLVED = -70
        self.EPISODE_REWARD_STD_SOLVED = 3


class ConfigMountainCar(ConfigGymClassicControl):
    def __init__(self):
        self.ENV_NAME = "MountainCar-v0"
        self.EPISODE_REWARD_AVG_SOLVED = -110
        self.EPISODE_REWARD_STD_SOLVED = 10
        self.N_TEST_EPISODES = 10
