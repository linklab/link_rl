class ConfigMujoco:
    pass


class ConfigAntMujoco(ConfigMujoco):
    def __init__(self):
        self.ENV_NAME = "Ant-v2"
        self.EPISODE_REWARD_MIN_SOLVED = 5000


class ConfigHopperMujoco(ConfigMujoco):
    def __init__(self):
        self.ENV_NAME = "Hopper-v2"
        self.EPISODE_REWARD_MIN_SOLVED = 3000


class ConfigWalker2dMujoco(ConfigMujoco):
    def __init__(self):
        self.ENV_NAME = "Walker2d-v2"
        self.EPISODE_REWARD_MIN_SOLVED = 5000


class ConfigHalfCheetahMujoco(ConfigMujoco):
    def __init__(self):
        self.ENV_NAME = "HalfCheetah-v2"
        self.EPISODE_REWARD_MIN_SOLVED = 12500


class ConfigInvertedDoublePendulumMujoco(ConfigMujoco):
    def __init__(self):
        self.ENV_NAME = "InvertedDoublePendulum-v2"
        self.EPISODE_REWARD_MIN_SOLVED = 8_500
