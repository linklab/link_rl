from a_configuration.a_base_config.a_environments.dm_control import ConfigDmControl
from g_utils.types import LossFunctionType


class ConfigDmControlCartpole(ConfigDmControl):
    def __init__(self):
        super(ConfigDmControl, self).__init__()
        self.DOMAIN_NAME = "cartpole"


class ConfigDmControlCartpoleBalance(ConfigDmControlCartpole):
    def __init__(self):
        super(ConfigDmControlCartpoleBalance, self).__init__()
        self.ENV_NAME = "dm_control_cartpole_balance"
        self.TASK_NAME = "balance"
        self.EPISODE_REWARD_AVG_SOLVED = 995.0
        self.EPISODE_REWARD_STD_SOLVED = 5.0


class ConfigDmControlCartpoleBalanceSparse(ConfigDmControlCartpole):
    def __init__(self):
        super(ConfigDmControlCartpoleBalanceSparse, self).__init__()
        self.ENV_NAME = "dm_control_cartpole_balance_sparse"
        self.TASK_NAME = "balance_sparse"
        self.EPISODE_REWARD_AVG_SOLVED = 450.0
        self.EPISODE_REWARD_STD_SOLVED = 50.0


class ConfigDmControlCartpoleSwingup(ConfigDmControlCartpole):
    def __init__(self):
        super(ConfigDmControlCartpoleSwingup, self).__init__()
        self.ENV_NAME = "dm_control_cartpole_swingup"
        self.TASK_NAME = "swingup"
        self.EPISODE_REWARD_AVG_SOLVED = 450.0
        self.EPISODE_REWARD_STD_SOLVED = 5.0


class ConfigDmControlCartpoleSwingupSparse(ConfigDmControlCartpole):
    def __init__(self):
        super(ConfigDmControlCartpoleSwingupSparse, self).__init__()
        self.ENV_NAME = "dm_control_cartpole_swingup_sparse"
        self.TASK_NAME = "swingup_sparse"
        self.EPISODE_REWARD_AVG_SOLVED = 450
        self.EPISODE_REWARD_STD_SOLVED = 50.0


class ConfigDmControlCartpoleTwoPoles(ConfigDmControlCartpole):
    def __init__(self):
        super(ConfigDmControlCartpoleTwoPoles, self).__init__()
        self.ENV_NAME = "dm_control_cartpole_two_poles"
        self.TASK_NAME = "two_poles"
        self.EPISODE_REWARD_AVG_SOLVED = 150
        self.EPISODE_REWARD_STD_SOLVED = 10.0


class ConfigDmControlCartpoleThreePoles(ConfigDmControlCartpole):
    def __init__(self):
        super(ConfigDmControlCartpoleThreePoles, self).__init__()
        self.ENV_NAME = "dm_control_cartpole_three_poles"
        self.TASK_NAME = "three_poles"
        self.EPISODE_REWARD_AVG_SOLVED = 499
        self.EPISODE_REWARD_STD_SOLVED = 10.0
