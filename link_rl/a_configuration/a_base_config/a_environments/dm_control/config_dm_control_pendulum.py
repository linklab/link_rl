from link_rl.a_configuration.a_base_config.a_environments.dm_control import ConfigDmControl


class ConfigDmControlPendulumSwingup(ConfigDmControl):
    def __init__(self):
        super(ConfigDmControlPendulumSwingup, self).__init__()
        self.ENV_NAME = "dm_control_pendulum_swingup"
        self.DOMAIN_NAME = "pendulum"
        self.TASK_NAME = "swingup"
        self.EPISODE_REWARD_AVG_SOLVED = 450
        self.EPISODE_REWARD_STD_SOLVED = 50.0
