from link_rl.a_configuration.a_base_config.a_environments.dm_control import ConfigDmControl
from link_rl.c_encoders.a_encoder import ENCODER


class ConfigDmControlBallInCupCatch(ConfigDmControl):
    def __init__(self):
        super(ConfigDmControlBallInCupCatch, self).__init__()
        self.ENV_NAME = "dm_control_ball_in_cup_catch"
        self.DOMAIN_NAME = "ball_in_cup"
        self.TASK_NAME = "catch"
        self.EPISODE_REWARD_MEAN_SOLVED = 990
