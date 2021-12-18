# [NOTE] 바로 아래 CartPoleBulletEnv import 구문 삭제하지 말것
from pybullet_envs.bullet.cartpole_bullet import CartPoleBulletEnv


class ParameterCartPoleBullet:
    def __init__(self):
        self.ENV_NAME = "CartPoleBulletEnv-v1"
        self.EPISODE_REWARD_AVG_SOLVED = 190
        self.EPISODE_REWARD_STD_SOLVED = 20
        self.TEST_INTERVAL_TRAINING_STEPS = 1024


class ParameterAntBullet:
    def __init__(self):
        self.ENV_NAME = "AntBulletEnv-v0"
        self.EPISODE_REWARD_AVG_SOLVED = 2_000
        self.EPISODE_REWARD_STD_SOLVED = 100
