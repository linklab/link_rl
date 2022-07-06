from link_rl.a_configuration.a_base_config.a_environments.dm_control import ConfigDmControl


class ConfigDmControlHumanoidWalk(ConfigDmControl):
    def __init__(self):
        super(ConfigDmControlHumanoidWalk, self).__init__()
        self.ENV_NAME = "dm_control_humanoid_walk"
        self.DOMAIN_NAME = "humanoid"
        self.TASK_NAME = "walk"
        self.EPISODE_REWARD_MIN_SOLVED = 450
        self.ACTION_REPEAT = 2


class ConfigDmControlHumanoidStand(ConfigDmControl):
    def __init__(self):
        super(ConfigDmControlHumanoidStand, self).__init__()
        self.ENV_NAME = "dm_control_humanoid_stand"
        self.DOMAIN_NAME = "humanoid"
        self.TASK_NAME = "stand"
        self.EPISODE_REWARD_MIN_SOLVED = 450
        self.ACTION_REPEAT = 2


class ConfigDmControlHumanoidRun(ConfigDmControl):
    def __init__(self):
        super(ConfigDmControlHumanoidRun, self).__init__()
        self.ENV_NAME = "dm_control_humanoid_run"
        self.DOMAIN_NAME = "humanoid"
        self.TASK_NAME = "run"
        self.EPISODE_REWARD_MIN_SOLVED = 450
        self.ACTION_REPEAT = 2
