import torch.nn.functional as F

from g_utils.types import LossFunctionType


class ConfigGymClassicControl:
    pass


class ConfigCartPole(ConfigGymClassicControl):
    def __init__(self):
        self.ENV_NAME = "CartPole-v1"
        self.EPISODE_REWARD_AVG_SOLVED = 450
        self.EPISODE_REWARD_STD_SOLVED = 50.0
