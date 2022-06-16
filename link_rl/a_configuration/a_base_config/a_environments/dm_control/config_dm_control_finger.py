from link_rl.a_configuration.a_base_config.a_environments.dm_control import ConfigDmControl


class ConfigDmControlFingerSpin(ConfigDmControl):
    def __init__(self):
        super(ConfigDmControlFingerSpin, self).__init__()
        self.ENV_NAME = "dm_control_finger_spin"
        self.DOMAIN_NAME = "finger"
        self.TASK_NAME = "spin"
        self.EPISODE_REWARD_AVG_SOLVED = 900
        self.EPISODE_REWARD_STD_SOLVED = 50.0
        self.ACTION_REPEAT = 2
