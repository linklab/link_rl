class ConfigMujoco:
    pass


class ConfigAntMujoco(ConfigMujoco):
    def __init__(self):
        self.ENV_NAME = "Ant-v2"
        self.EPISODE_REWARD_AVG_SOLVED = 5000
        self.EPISODE_REWARD_STD_SOLVED = 300


class ConfigHopperMujoco(ConfigMujoco):
    def __init__(self):
        self.ENV_NAME = "Hopper-v2"
        self.EPISODE_REWARD_AVG_SOLVED = 3000
        self.EPISODE_REWARD_STD_SOLVED = 300


class ConfigWalker2dMujoco(ConfigMujoco):
    def __init__(self):
        self.ENV_NAME = "Walker2d-v2"
        self.EPISODE_REWARD_AVG_SOLVED = 5000
        self.EPISODE_REWARD_STD_SOLVED = 300


class ConfigHalfCheetahMujoco(ConfigMujoco):
    def __init__(self):
        self.ENV_NAME = "HalfCheetah-v2"
        self.EPISODE_REWARD_AVG_SOLVED = 12500
        self.EPISODE_REWARD_STD_SOLVED = 300


class ConfigInvertedDoublePendulumMujoco(ConfigMujoco):
    def __init__(self):
        self.ENV_NAME = "InvertedDoublePendulum-v2"
        self.EPISODE_REWARD_AVG_SOLVED = 8_500
        self.EPISODE_REWARD_STD_SOLVED = 500