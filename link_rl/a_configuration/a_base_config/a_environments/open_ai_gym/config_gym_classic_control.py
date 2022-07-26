from link_rl.a_configuration.a_base_config.a_environments.open_ai_gym import ConfigGymClassicControl


class ConfigCartPole(ConfigGymClassicControl):
    def __init__(self):
        super(ConfigCartPole, self).__init__()
        self.ENV_NAME = "CartPole-v1"
        self.EPISODE_REWARD_MEAN_SOLVED = 450


class ConfigAcrobot(ConfigGymClassicControl):
    def __init__(self):
        super(ConfigAcrobot, self).__init__()
        self.ENV_NAME = "Acrobot-v1"
        self.EPISODE_REWARD_MEAN_SOLVED = -70


class ConfigMountainCar(ConfigGymClassicControl):
    def __init__(self):
        super(ConfigMountainCar, self).__init__()
        self.ENV_NAME = "MountainCar-v0"
        self.EPISODE_REWARD_MEAN_SOLVED = -110
        self.N_TEST_EPISODES = 10
