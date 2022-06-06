from a_configuration.a_base_config.a_environments.dm_control import ConfigDmControl
from g_utils.types import LossFunctionType


class ConfigDmControlCheetahRun(ConfigDmControl):
    def __init__(self):
        super(ConfigDmControl, self).__init__()
        self.ENV_NAME = "dm_control_cheetah_run"
        self.DOMAIN_NAME = "cheetah"
        self.TASK_NAME = "run"
        self.EPISODE_REWARD_AVG_SOLVED = 500
        self.EPISODE_REWARD_STD_SOLVED = 50.0
