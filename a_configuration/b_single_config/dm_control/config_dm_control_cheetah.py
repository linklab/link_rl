from a_configuration.a_base_config.a_environments.dm_control.config_dm_control_cheetah import ConfigDmControlCheetahRun
from a_configuration.a_base_config.b_agents.config_agents_off_policy import ConfigDdpg, ConfigTd3, ConfigSac
from a_configuration.a_base_config.b_agents.config_agents_on_policy import ConfigA2c, ConfigPpo, ConfigA3c
from a_configuration.a_base_config.config_single_base import ConfigBase
from g_utils.types import ModelType


class ConfigDmControlCheetahA2c(ConfigBase, ConfigDmControlCheetahRun, ConfigA2c):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigDmControlCheetahRun.__init__(self)
        ConfigA2c.__init__(self)

        self.MAX_TRAINING_STEPS = 500_000
        self.MODEL_TYPE = ModelType.SMALL_LINEAR


class ConfigDmControlCheetahA3c(ConfigBase, ConfigDmControlCheetahRun, ConfigA3c):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigDmControlCheetahRun.__init__(self)
        ConfigA3c.__init__(self)

        self.MAX_TRAINING_STEPS = 500_000
        self.MODEL_TYPE = ModelType.SMALL_LINEAR


class ConfigDmControlCheetahPpo(ConfigBase, ConfigDmControlCheetahRun, ConfigPpo):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigDmControlCheetahRun.__init__(self)
        ConfigPpo.__init__(self)

        self.MAX_TRAINING_STEPS = 500_000
        self.MODEL_TYPE = ModelType.SMALL_LINEAR


class ConfigDmControlCheetahDdpg(ConfigBase, ConfigDmControlCheetahRun, ConfigDdpg):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigDmControlCheetahRun.__init__(self)
        ConfigDdpg.__init__(self)

        self.MAX_TRAINING_STEPS = 500_000
        self.MODEL_TYPE = ModelType.SMALL_LINEAR


class ConfigDmControlCheetahTd3(ConfigBase, ConfigDmControlCheetahRun, ConfigTd3):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigDmControlCheetahRun.__init__(self)
        ConfigTd3.__init__(self)

        self.MAX_TRAINING_STEPS = 500_000
        self.MODEL_TYPE = ModelType.SMALL_LINEAR


class ConfigDmControlCheetahSac(ConfigBase, ConfigDmControlCheetahRun, ConfigSac):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigDmControlCheetahRun.__init__(self)
        ConfigSac.__init__(self)

        self.MAX_TRAINING_STEPS = 500_000
        self.MODEL_TYPE = ModelType.SMALL_LINEAR
