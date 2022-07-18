from link_rl.a_configuration.a_base_config.a_environments.evolution_gym import ConfigEvolutionGym


class ConfigEvolutionGymWalker(ConfigEvolutionGym):
    def __init__(self):
        super(ConfigEvolutionGymWalker, self).__init__()
        self.ENV_NAME = "Walker-v0"
        self.EPISODE_REWARD_MIN_SOLVED = 1_000


class ConfigEvolutionGymBridgeWalker(ConfigEvolutionGym):
    def __init__(self):
        super(ConfigEvolutionGymBridgeWalker, self).__init__()
        self.ENV_NAME = "BridgeWalker-v0"
        self.EPISODE_REWARD_MIN_SOLVED = 1_000


class ConfigEvolutionGymCaveCrawler(ConfigEvolutionGym):
    def __init__(self):
        super(ConfigEvolutionGymCaveCrawler, self).__init__()
        self.ENV_NAME = "CaveCrawler-v0"
        self.EPISODE_REWARD_MIN_SOLVED = 1_000
