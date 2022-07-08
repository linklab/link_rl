from link_rl.a_configuration.a_base_config.b_agents.config_agents_off_policy import ConfigDdpg
from link_rl.a_configuration.a_base_config.config_single_base import ConfigBase
from link_rl.a_configuration.a_base_config.a_environments.unity.config_unity_box import Config3DBall
from link_rl.c_models_v2.e_ddpg_model import DDPG_MODEL


class Config3DBallDdqg(ConfigBase, Config3DBall, ConfigDdpg):
    def __init__(self):
        ConfigBase.__init__(self)
        Config3DBall.__init__(self)
        ConfigDdpg.__init__(self)

        self.N_STEP = 1
        self.BUFFER_CAPACITY = 250_000

        self.MAX_TRAINING_STEPS = 2_000_000
        self.MODEL_TYPE = DDPG_MODEL.ContinuousDdpgModel.value
