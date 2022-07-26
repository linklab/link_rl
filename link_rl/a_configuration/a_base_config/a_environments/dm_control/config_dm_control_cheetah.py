from link_rl.a_configuration.a_base_config.a_environments.dm_control import ConfigDmControl


class ConfigDmControlCheetahRun(ConfigDmControl):
    def __init__(self):
        super(ConfigDmControlCheetahRun, self).__init__()
        self.ENV_NAME = "dm_control_cheetah_run"
        self.DOMAIN_NAME = "cheetah"
        self.TASK_NAME = "run"
        self.EPISODE_REWARD_MEAN_SOLVED = 850
