# [NOTE] 바로 아래 CartPoleBulletEnv import 구문 삭제하지 말것


class ConfigBullet:
    pass


class ConfigCartPoleBullet(ConfigBullet):
    def __init__(self):
        self.ENV_NAME = "CartPoleBulletEnv-v1"
        self.EPISODE_REWARD_MIN_SOLVED = 190


class ConfigCartPoleContinuousBullet(ConfigBullet):
    def __init__(self):
        self.ENV_NAME = "CartPoleContinuousBulletEnv-v0"
        self.EPISODE_REWARD_MIN_SOLVED = 190


class ConfigAntBullet(ConfigBullet):
    def __init__(self):
        self.ENV_NAME = "AntBulletEnv-v0"
        self.EPISODE_REWARD_MIN_SOLVED = 2_000


class ConfigHopperBullet(ConfigBullet):
    def __init__(self):
        self.ENV_NAME = "HopperBulletEnv-v0"
        self.EPISODE_REWARD_MIN_SOLVED = 2_000


class ConfigInvertedDoublePendulumBullet(ConfigBullet):
    def __init__(self):
        self.ENV_NAME = "InvertedDoublePendulumBulletEnv-v0"
        self.EPISODE_REWARD_MIN_SOLVED = 8_500
        self.TEST_INTERVAL_TRAINING_STEPS = 5_000


class ConfigHumanoidBullet(ConfigBullet):
    def __init__(self):
        self.ENV_NAME = "HumanoidBulletEnv-v0"
        self.EPISODE_REWARD_MIN_SOLVED = 3_000
        self.TEST_INTERVAL_TRAINING_STEPS = 5_000

