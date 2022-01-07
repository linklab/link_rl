# [NOTE] 바로 아래 CartPoleBulletEnv import 구문 삭제하지 말것
from pybullet_envs.bullet.cartpole_bullet import CartPoleBulletEnv


class ParameterBullet:
    pass


class ParameterCartPoleBullet(ParameterBullet):
    def __init__(self):
        self.ENV_NAME = "CartPoleBulletEnv-v1"
        self.EPISODE_REWARD_AVG_SOLVED = 190
        self.EPISODE_REWARD_STD_SOLVED = 20
        self.TEST_INTERVAL_TRAINING_STEPS = 1_024


class ParameterCartPoleContinuousBullet(ParameterBullet):
    def __init__(self):
        self.ENV_NAME = "CartPoleContinuousBulletEnv-v0"
        self.EPISODE_REWARD_AVG_SOLVED = 190
        self.EPISODE_REWARD_STD_SOLVED = 20
        self.TEST_INTERVAL_TRAINING_STEPS = 1_024


class ParameterAntBullet(ParameterBullet):
    def __init__(self):
        self.ENV_NAME = "AntBulletEnv-v0"
        self.EPISODE_REWARD_AVG_SOLVED = 2_000
        self.EPISODE_REWARD_STD_SOLVED = 100
        self.TEST_INTERVAL_TRAINING_STEPS = 5_000


class ParameterHopperBullet(ParameterBullet):
    def __init__(self):
        self.ENV_NAME = "HopperBulletEnv-v0"
        self.EPISODE_REWARD_AVG_SOLVED = 2_000
        self.EPISODE_REWARD_STD_SOLVED = 100
        self.TEST_INTERVAL_TRAINING_STEPS = 5_000


class ParameterDoubleInvertedPendulumBullet(ParameterBullet):
    def __init__(self):
        self.ENV_NAME = "InvertedDoublePendulumBulletEnv-v0"
        self.EPISODE_REWARD_AVG_SOLVED = 9_100
        self.EPISODE_REWARD_STD_SOLVED = 500
        self.TEST_INTERVAL_TRAINING_STEPS = 5_000
