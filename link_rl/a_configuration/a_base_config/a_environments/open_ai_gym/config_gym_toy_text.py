from link_rl.a_configuration.a_base_config.a_environments.open_ai_gym import ConfigGymToyText


class ConfigFrozenLake(ConfigGymToyText):
    def __init__(self):
        super(ConfigFrozenLake, self).__init__()
        self.ENV_NAME = "FrozenLake-v1"
        self.EPISODE_REWARD_MIN_SOLVED = 80
        self.N_TEST_EPISODES = 40
        self.RANDOM_MAP = False
        self.BOX_OBSERVATION = False
