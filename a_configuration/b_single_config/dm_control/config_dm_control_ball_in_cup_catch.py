from a_configuration.a_base_config.a_environments.dm_control.config_dm_control_ball_in_cup import \
    ConfigDmControlBallInCupCatch
from a_configuration.a_base_config.b_agents.config_agents_off_policy import ConfigDdpg, ConfigTd3, ConfigSac
from a_configuration.a_base_config.b_agents.config_agents_on_policy import ConfigA2c, ConfigPpo, ConfigA3c
from a_configuration.a_base_config.config_single_base import ConfigBase
from g_utils.types import ModelType


class ConfigDmControlBallInCupCatchA2c(ConfigBase, ConfigDmControlBallInCupCatch, ConfigA2c):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigDmControlBallInCupCatch.__init__(self)
        ConfigA2c.__init__(self)

        self.MAX_TRAINING_STEPS = 1_000_000
        self.MODEL_TYPE = ModelType.SMALL_LINEAR


class ConfigDmControlBallInCupCatchA3c(ConfigBase, ConfigDmControlBallInCupCatch, ConfigA3c):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigDmControlBallInCupCatch.__init__(self)
        ConfigA3c.__init__(self)

        self.MAX_TRAINING_STEPS = 1_000_000
        self.MODEL_TYPE = ModelType.SMALL_LINEAR


class ConfigDmControlBallInCupCatchPpo(ConfigBase, ConfigDmControlBallInCupCatch, ConfigPpo):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigDmControlBallInCupCatch.__init__(self)
        ConfigPpo.__init__(self)

        self.MAX_TRAINING_STEPS = 1_000_000
        self.MODEL_TYPE = ModelType.SMALL_LINEAR


class ConfigDmControlBallInCupCatchDdpg(ConfigBase, ConfigDmControlBallInCupCatch, ConfigDdpg):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigDmControlBallInCupCatch.__init__(self)
        ConfigDdpg.__init__(self)

        self.MAX_TRAINING_STEPS = 1_000_000
        self.MODEL_TYPE = ModelType.SMALL_LINEAR


class ConfigDmControlBallInCupCatchTd3(ConfigBase, ConfigDmControlBallInCupCatch, ConfigTd3):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigDmControlBallInCupCatch.__init__(self)
        ConfigTd3.__init__(self)

        self.MAX_TRAINING_STEPS = 1_000_000
        self.MODEL_TYPE = ModelType.SMALL_LINEAR


class ConfigDmControlBallInCupCatchSac(ConfigBase, ConfigDmControlBallInCupCatch, ConfigSac):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigDmControlBallInCupCatch.__init__(self)
        ConfigSac.__init__(self)

        self.MAX_TRAINING_STEPS = 1_000_000
        self.MODEL_TYPE = ModelType.SMALL_LINEAR
