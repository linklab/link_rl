from link_rl.a_configuration.a_base_config.a_environments.somo_gym import ConfigSomoGym


class ConfigSomoGymInHandManipulation(ConfigSomoGym):
    def __init__(self):
        self.ENV_NAME = "InHandManipulation"
        self.EPISODE_REWARD_MIN_SOLVED = 2_000
        super(ConfigSomoGymInHandManipulation, self).__init__(self.ENV_NAME)
