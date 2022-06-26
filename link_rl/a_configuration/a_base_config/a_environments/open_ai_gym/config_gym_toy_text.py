from link_rl.g_utils.types import LossFunctionType


class ConfigGymToyText:
    pass


class ConfigFrozenLake(ConfigGymToyText):
    def __init__(self):
        self.ENV_NAME = "FrozenLake-v1"
        self.EPISODE_REWARD_MIN_SOLVED = 80
        self.N_TEST_EPISODES = 40
        self.RANDOM_MAP = False
        self.BOX_OBSERVATION = False

        self.LOSS_FUNCTION_TYPE = LossFunctionType.MSE_LOSS
