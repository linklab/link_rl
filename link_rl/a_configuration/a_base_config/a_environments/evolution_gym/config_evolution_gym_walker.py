from link_rl.a_configuration.a_base_config.a_environments.evolution_gym import ConfigEvolutionGym
import numpy as np

class ConfigEvolutionGymWalker(ConfigEvolutionGym):
    def __init__(self):
        super(ConfigEvolutionGymWalker, self).__init__()
        self.ENV_NAME = "Walker-v0"
        self.EPISODE_REWARD_MEAN_SOLVED = 1_000

        self.ROBOT_SHAPE = (3, 3)
        # self.ROBOT_STRUCTURE, self.ROBOT_CONNECTIONS = sample_robot(self.ROBOT_SHAPE)

        self.ROBOT_STRUCTURE = np.asarray([
            [3, 3, 3],
            [3, 0, 3],
            [3, 0, 3]
        ])


class ConfigEvolutionGymBridgeWalker(ConfigEvolutionGym):
    def __init__(self):
        super(ConfigEvolutionGymBridgeWalker, self).__init__()
        self.ENV_NAME = "BridgeWalker-v0"
        self.EPISODE_REWARD_MEAN_SOLVED = 1_000


class ConfigEvolutionGymCaveCrawler(ConfigEvolutionGym):
    def __init__(self):
        super(ConfigEvolutionGymCaveCrawler, self).__init__()
        self.ENV_NAME = "CaveCrawler-v0"
        self.EPISODE_REWARD_MEAN_SOLVED = 1_000
