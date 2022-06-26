from link_rl.a_configuration.a_base_config.a_environments.dm_control import ConfigDmControl


class ConfigDmControlCartpole(ConfigDmControl):
    def __init__(self):
        super(ConfigDmControlCartpole, self).__init__()
        self.DOMAIN_NAME = "cartpole"
        self.TASK_NAME = None


class ConfigDmControlCartpoleBalance(ConfigDmControlCartpole):
    def __init__(self):
        super(ConfigDmControlCartpoleBalance, self).__init__()
        self.ENV_NAME = "dm_control_cartpole_balance"
        self.TASK_NAME = "balance"
        self.EPISODE_REWARD_MIN_SOLVED = 995.0


class ConfigDmControlCartpoleBalanceSparse(ConfigDmControlCartpole):
    def __init__(self):
        super(ConfigDmControlCartpoleBalanceSparse, self).__init__()
        self.ENV_NAME = "dm_control_cartpole_balance_sparse"
        self.TASK_NAME = "balance_sparse"
        self.EPISODE_REWARD_MIN_SOLVED = 450.0


class ConfigDmControlCartpoleSwingup(ConfigDmControlCartpole):
    def __init__(self):
        super(ConfigDmControlCartpoleSwingup, self).__init__()
        self.ENV_NAME = "dm_control_cartpole_swingup"
        self.TASK_NAME = "swingup"
        self.EPISODE_REWARD_MIN_SOLVED = 450.0


class ConfigDmControlCartpoleSwingupSparse(ConfigDmControlCartpole):
    def __init__(self):
        super(ConfigDmControlCartpoleSwingupSparse, self).__init__()
        self.ENV_NAME = "dm_control_cartpole_swingup_sparse"
        self.TASK_NAME = "swingup_sparse"
        self.EPISODE_REWARD_MIN_SOLVED = 450


class ConfigDmControlCartpoleTwoPoles(ConfigDmControlCartpole):
    def __init__(self):
        super(ConfigDmControlCartpoleTwoPoles, self).__init__()
        self.ENV_NAME = "dm_control_cartpole_two_poles"
        self.TASK_NAME = "two_poles"
        self.EPISODE_REWARD_MIN_SOLVED = 150


class ConfigDmControlCartpoleThreePoles(ConfigDmControlCartpole):
    def __init__(self):
        super(ConfigDmControlCartpoleThreePoles, self).__init__()
        self.ENV_NAME = "dm_control_cartpole_three_poles"
        self.TASK_NAME = "three_poles"
        self.EPISODE_REWARD_MIN_SOLVED = 499
