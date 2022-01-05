class ParameterGymClassicControl:
    pass


class ParameterCartPole(ParameterGymClassicControl):
    def __init__(self):
        self.ENV_NAME = "CartPole-v1"
        self.EPISODE_REWARD_AVG_SOLVED = 450
        self.EPISODE_REWARD_STD_SOLVED = 50.0
        self.TEST_INTERVAL_TRAINING_STEPS = 1024
