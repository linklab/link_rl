from a_configuration.a_base_config.a_environments.dm_control.config_dm_control_cartpole import \
    ConfigDmControlCartpoleBalance
from a_configuration.a_base_config.b_agents.config_agents_on_policy import ConfigA2c, ConfigPpo, ConfigA3c
from a_configuration.a_base_config.config_single_base import ConfigBase
from g_utils.types import ModelType


class ConfigDmControlCartPoleBalanceA2c(ConfigBase, ConfigDmControlCartpoleBalance, ConfigA2c):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigDmControlCartpoleBalance.__init__(self)
        ConfigA2c.__init__(self)

        self.MAX_TRAINING_STEPS = 100_000
        self.MODEL_TYPE = ModelType.SMALL_LINEAR


class ConfigDmControlCartPoleBalanceA3c(ConfigBase, ConfigDmControlCartpoleBalance, ConfigA3c):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigDmControlCartpoleBalance.__init__(self)
        ConfigA3c.__init__(self)

        self.MAX_TRAINING_STEPS = 100_000
        self.MODEL_TYPE = ModelType.SMALL_LINEAR


class ConfigDmControlCartPoleBalancePpo(ConfigBase, ConfigDmControlCartpoleBalance, ConfigPpo):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigDmControlCartpoleBalance.__init__(self)
        ConfigPpo.__init__(self)

        self.MAX_TRAINING_STEPS = 100_000
        self.MODEL_TYPE = ModelType.SMALL_LINEAR



