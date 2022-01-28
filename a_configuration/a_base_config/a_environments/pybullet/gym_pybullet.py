# [NOTE] 바로 아래 CartPoleBulletEnv import 구문 삭제하지 말것
from pybullet_envs.bullet.cartpole_bullet import CartPoleBulletEnv
import torch.nn.functional as F

from g_utils.types import LossFunctionType


class ConfigBullet:
    pass


class ConfigCartPoleBullet(ConfigBullet):
    def __init__(self):
        self.ENV_NAME = "CartPoleBulletEnv-v1"
        self.EPISODE_REWARD_AVG_SOLVED = 190
        self.EPISODE_REWARD_STD_SOLVED = 20

        self.LOSS_FUNCTION_TYPE = LossFunctionType.MSE_LOSS


class ConfigCartPoleContinuousBullet(ConfigBullet):
    def __init__(self):
        self.ENV_NAME = "CartPoleContinuousBulletEnv-v0"
        self.EPISODE_REWARD_AVG_SOLVED = 190
        self.EPISODE_REWARD_STD_SOLVED = 20

        self.LOSS_FUNCTION_TYPE = LossFunctionType.MSE_LOSS


class ConfigAntBullet(ConfigBullet):
    def __init__(self):
        self.ENV_NAME = "AntBulletEnv-v0"
        self.EPISODE_REWARD_AVG_SOLVED = 2_000
        self.EPISODE_REWARD_STD_SOLVED = 100


class ConfigHopperBullet(ConfigBullet):
    def __init__(self):
        self.ENV_NAME = "HopperBulletEnv-v0"
        self.EPISODE_REWARD_AVG_SOLVED = 2_000
        self.EPISODE_REWARD_STD_SOLVED = 100


class ConfigDoubleInvertedPendulumBullet(ConfigBullet):
    def __init__(self):
        self.ENV_NAME = "InvertedDoublePendulumBulletEnv-v0"
        self.EPISODE_REWARD_AVG_SOLVED = 8_500
        self.EPISODE_REWARD_STD_SOLVED = 500
