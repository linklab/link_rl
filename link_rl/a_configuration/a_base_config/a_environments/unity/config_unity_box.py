class ConfigUnityGymEnv:
    pass


class Config3DBall(ConfigUnityGymEnv):
    def __init__(self):
        self.ENV_NAME = "Unity3DBall"
        self.EPISODE_REWARD_MEAN_SOLVED = 100
        self.NO_TEST_GRAPHICS = True
        self.width = 600
        self.height = 600
        self.time_scale = 2.0


class ConfigWalker(ConfigUnityGymEnv):
    def __init__(self):
        self.ENV_NAME = "UnityWalker"
        self.EPISODE_REWARD_MEAN_SOLVED = 500
        self.NO_TEST_GRAPHICS = True
        self.width = 600
        self.height = 600
        self.time_scale = 2.0


class ConfigDrone(ConfigUnityGymEnv):
    def __init__(self):
        self.ENV_NAME = "UnityDrone"
        self.EPISODE_REWARD_MEAN_SOLVED = 200
        self.NO_TEST_GRAPHICS = True
        self.width = 600
        self.height = 600
        self.time_scale = 12.0
