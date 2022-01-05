class ParameterMujoco:
    pass


class ParameterAntMujoco(ParameterMujoco):
    def __init__(self):
        self.ENV_NAME = "Ant-v2"
        self.EPISODE_REWARD_AVG_SOLVED = 5000
        self.EPISODE_REWARD_STD_SOLVED = 300
        self.TEST_INTERVAL_TRAINING_STEPS = 1_024


class ParameterHopperMujoco(ParameterMujoco):
    def __init__(self):
        self.ENV_NAME = "Hopper-v2"
        self.EPISODE_REWARD_AVG_SOLVED = 3000
        self.EPISODE_REWARD_STD_SOLVED = 300
        self.TEST_INTERVAL_TRAINING_STEPS = 1_024


class ParameterWalker2dMujoco(ParameterMujoco):
    def __init__(self):
        self.ENV_NAME = "Walker2d-v2"
        self.EPISODE_REWARD_AVG_SOLVED = 5000
        self.EPISODE_REWARD_STD_SOLVED = 300
        self.TEST_INTERVAL_TRAINING_STEPS = 1_024


class ParameterHalfCheetahMujoco(ParameterMujoco):
    def __init__(self):
        self.ENV_NAME = "HalfCheetah-v2"
        self.EPISODE_REWARD_AVG_SOLVED = 12500
        self.EPISODE_REWARD_STD_SOLVED = 300
        self.TEST_INTERVAL_TRAINING_STEPS = 1_024