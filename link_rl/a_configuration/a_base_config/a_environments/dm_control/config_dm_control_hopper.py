from link_rl.a_configuration.a_base_config.a_environments.dm_control import ConfigDmControl


class ConfigDmControlHopperHop(ConfigDmControl):
    def __init__(self):
        super(ConfigDmControlHopperHop, self).__init__()
        self.ENV_NAME = "dm_control_hopper_hop"
        self.DOMAIN_NAME = "hopper"
        self.TASK_NAME = "hop"
        self.EPISODE_REWARD_MEAN_SOLVED = 460
        self.ACTION_REPEAT = 4


class ConfigDmControlHopperStand(ConfigDmControl):
    def __init__(self):
        super(ConfigDmControlHopperStand, self).__init__()
        self.ENV_NAME = "dm_control_hopper_stand"
        self.DOMAIN_NAME = "hopper"
        self.TASK_NAME = "stand"
        self.EPISODE_REWARD_MEAN_SOLVED = 1000
        self.ACTION_REPEAT = 4
