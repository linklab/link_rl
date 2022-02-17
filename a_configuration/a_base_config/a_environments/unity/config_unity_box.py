class ConfigUnityGymEnv:
    pass


class Config3DBall(ConfigUnityGymEnv):
    def __init__(self):
        self.ENV_NAME = "Unity3DBall"
        self.EPISODE_REWARD_AVG_SOLVED = 100
        self.EPISODE_REWARD_STD_SOLVED = 1
        self.NO_TEST_GRAPHICS = True
        self.width = 600
        self.height = 600
        self.time_scale = 2.0


class ConfigWalker(ConfigUnityGymEnv):
    def __init__(self):
        self.ENV_NAME = "UnityWalker"
        self.EPISODE_REWARD_AVG_SOLVED = 500
        self.EPISODE_REWARD_STD_SOLVED = 500
        self.NO_TEST_GRAPHICS = True
        self.width = 600
        self.height = 600
        self.time_scale = 2.0


class ConfigDrone(ConfigUnityGymEnv):
    def __init__(self):
        self.ENV_NAME = "UnityDrone"
        self.EPISODE_REWARD_AVG_SOLVED = 200
        self.EPISODE_REWARD_STD_SOLVED = 50
        self.NO_TEST_GRAPHICS = True
        self.width = 600
        self.height = 600
        self.time_scale = 2.0
