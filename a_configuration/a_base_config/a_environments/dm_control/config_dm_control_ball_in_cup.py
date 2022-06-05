from a_configuration.a_base_config.a_environments.dm_control import ConfigDmControl
from g_utils.types import LossFunctionType


class ConfigDmControlBallInCupCatch(ConfigDmControl):
    def __init__(self):
        super(ConfigDmControl, self).__init__()
        self.ENV_NAME = "dm_control_ball_in_cup_catch"
        self.DOMAIN_NAME = "ball_in_cup"
        self.TASK_NAME = "catch"
        self.EPISODE_REWARD_AVG_SOLVED = 450
        self.EPISODE_REWARD_STD_SOLVED = 50.0
