class ConfigUnityGymEnv:
    pass


class Config3DBall(ConfigUnityGymEnv):
    def __init__(self):
        self.ENV_NAME = "Unity3DBall"
        self.EPISODE_REWARD_AVG_SOLVED = 100
        self.EPISODE_REWARD_STD_SOLVED = 1
        self.NO_TEST_GRAPHICS = True
        self.time_scale = 2.0

class ConfigWalker(ConfigUnityGymEnv):
    def __init__(self):
        self.ENV_NAME = "UnityWalker"
        self.EPISODE_REWARD_AVG_SOLVED = 500
        self.EPISODE_REWARD_STD_SOLVED = 500
        self.NO_TEST_GRAPHICS = True
        self.time_scale = 2.0
