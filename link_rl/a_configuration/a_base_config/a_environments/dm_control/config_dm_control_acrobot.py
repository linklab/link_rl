from link_rl.a_configuration.a_base_config.a_environments.dm_control import ConfigDmControl


class ConfigDmControlAcrobotSwingUp(ConfigDmControl):
    def __init__(self):
        super(ConfigDmControlAcrobotSwingUp, self).__init__()
        self.ENV_NAME = "dm_control_acrobot_swing_up"
        self.DOMAIN_NAME = "acrobot"
        self.TASK_NAME = "swingup"
        self.EPISODE_REWARD_MEAN_SOLVED = 400
        self.ACTION_REPEAT = 4


class ConfigDmControlAcrobotSwingUpSparse(ConfigDmControl):
    def __init__(self):
        super(ConfigDmControlAcrobotSwingUpSparse, self).__init__()
        self.ENV_NAME = "dm_control_acrobot_swing_up_sparse"
        self.DOMAIN_NAME = "acrobot"
        self.TASK_NAME = "swingup_sparse"
        self.EPISODE_REWARD_MEAN_SOLVED = 300
        self.ACTION_REPEAT = 4
