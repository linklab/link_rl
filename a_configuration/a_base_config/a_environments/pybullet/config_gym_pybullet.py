# [NOTE] 바로 아래 CartPoleBulletEnv import 구문 삭제하지 말것
import torch.nn.functional as F

from g_utils.types import LossFunctionType


class ConfigBullet:
    pass


class ConfigCartPoleBullet(ConfigBullet):
    def __init__(self):
        from pybullet_envs.bullet.cartpole_bullet import CartPoleBulletEnv
        self.ENV_NAME = "CartPoleBulletEnv-v1"
        self.EPISODE_REWARD_AVG_SOLVED = 190
        self.EPISODE_REWARD_STD_SOLVED = 20


class ConfigCartPoleContinuousBullet(ConfigBullet):
    def __init__(self):
        from pybullet_envs.bullet.cartpole_bullet import CartPoleBulletEnv
        self.ENV_NAME = "CartPoleContinuousBulletEnv-v0"
        self.EPISODE_REWARD_AVG_SOLVED = 190
        self.EPISODE_REWARD_STD_SOLVED = 20


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


class ConfigInvertedDoublePendulumBullet(ConfigBullet):
    def __init__(self):
        self.ENV_NAME = "InvertedDoublePendulumBulletEnv-v0"
        self.EPISODE_REWARD_AVG_SOLVED = 8_500
        self.EPISODE_REWARD_STD_SOLVED = 500
        self.TEST_INTERVAL_TRAINING_STEPS = 5_000


class ConfigHumanoidBullet(ConfigBullet):
    def __init__(self):
        self.ENV_NAME = "HumanoidBulletEnv-v0"
        self.EPISODE_REWARD_AVG_SOLVED = 3_000
        self.EPISODE_REWARD_STD_SOLVED = 100
        self.TEST_INTERVAL_TRAINING_STEPS = 5_000

