from a_configuration.a_base_config.a_environments.dm_control.config_dm_control_cartpole import \
    ConfigDmControlCartpoleBalance, ConfigDmControlCartpoleThreePoles
from a_configuration.a_base_config.b_agents.config_agents_off_policy import ConfigDdpg, ConfigTd3, ConfigSac
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


class ConfigDmControlCartPoleBalanceDdpg(ConfigBase, ConfigDmControlCartpoleBalance, ConfigDdpg):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigDmControlCartpoleBalance.__init__(self)
        ConfigDdpg.__init__(self)

        self.MAX_TRAINING_STEPS = 100_000
        self.MODEL_TYPE = ModelType.SMALL_LINEAR


class ConfigDmControlCartPoleBalanceTd3(ConfigBase, ConfigDmControlCartpoleBalance, ConfigTd3):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigDmControlCartpoleBalance.__init__(self)
        ConfigTd3.__init__(self)

        self.MAX_TRAINING_STEPS = 100_000
        self.MODEL_TYPE = ModelType.SMALL_LINEAR


class ConfigDmControlCartPoleBalanceSac(ConfigBase, ConfigDmControlCartpoleBalance, ConfigSac):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigDmControlCartpoleBalance.__init__(self)
        ConfigSac.__init__(self)

        self.MAX_TRAINING_STEPS = 100_000
        self.MODEL_TYPE = ModelType.SMALL_LINEAR


### Three Poles


class ConfigDmControlCartPoleThreePolesA2c(ConfigBase, ConfigDmControlCartpoleThreePoles, ConfigA2c):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigDmControlCartpoleThreePoles.__init__(self)
        ConfigA2c.__init__(self)

        self.MAX_TRAINING_STEPS = 100_000
        self.MODEL_TYPE = ModelType.SMALL_LINEAR


class ConfigDmControlCartPoleThreePolesA3c(ConfigBase, ConfigDmControlCartpoleThreePoles, ConfigA3c):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigDmControlCartpoleThreePoles.__init__(self)
        ConfigA3c.__init__(self)

        self.MAX_TRAINING_STEPS = 100_000
        self.MODEL_TYPE = ModelType.SMALL_LINEAR


class ConfigDmControlCartPoleThreePolesPpo(ConfigBase, ConfigDmControlCartpoleThreePoles, ConfigPpo):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigDmControlCartpoleThreePoles.__init__(self)
        ConfigPpo.__init__(self)

        self.MAX_TRAINING_STEPS = 100_000
        self.MODEL_TYPE = ModelType.SMALL_LINEAR


class ConfigDmControlCartPoleThreePolesDdpg(ConfigBase, ConfigDmControlCartpoleThreePoles, ConfigDdpg):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigDmControlCartpoleThreePoles.__init__(self)
        ConfigDdpg.__init__(self)

        self.MAX_TRAINING_STEPS = 100_000
        self.MODEL_TYPE = ModelType.SMALL_LINEAR


class ConfigDmControlCartPoleThreePolesTd3(ConfigBase, ConfigDmControlCartpoleThreePoles, ConfigTd3):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigDmControlCartpoleThreePoles.__init__(self)
        ConfigTd3.__init__(self)

        self.MAX_TRAINING_STEPS = 100_000
        self.MODEL_TYPE = ModelType.SMALL_LINEAR


class ConfigDmControlCartPoleThreePolesSac(ConfigBase, ConfigDmControlCartpoleThreePoles, ConfigSac):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigDmControlCartpoleThreePoles.__init__(self)
        ConfigSac.__init__(self)

        self.MAX_TRAINING_STEPS = 100_000
        self.MODEL_TYPE = ModelType.SMALL_LINEAR