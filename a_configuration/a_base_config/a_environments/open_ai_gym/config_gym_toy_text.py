import torch.nn.functional as F

from g_utils.types import LossFunctionType


class ConfigGymToyText:
    pass


class ConfigFrozenLake(ConfigGymToyText):
    def __init__(self):
        self.ENV_NAME = "FrozenLake-v1"
        self.EPISODE_REWARD_AVG_SOLVED = 80
        self.EPISODE_REWARD_STD_SOLVED = 20
        self.N_TEST_EPISODES = 40
        self.RANDOM_MAP = False

        self.LOSS_FUNCTION_TYPE = LossFunctionType.MSE_LOSS
