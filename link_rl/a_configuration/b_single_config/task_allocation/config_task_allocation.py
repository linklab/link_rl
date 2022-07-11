from link_rl.a_configuration.a_base_config.a_environments.task_allocation.config_task_allocation import \
    ConfigTaskAllocationInitParam
from link_rl.a_configuration.a_base_config.b_agents.config_agents_off_policy import ConfigDqn, ConfigDoubleDqn
from link_rl.a_configuration.a_base_config.config_single_base import ConfigBase
from link_rl.d_models.b_q_model import Q_MODEL


class ConfigTaskAllocationDqn(ConfigBase, ConfigTaskAllocationInitParam, ConfigDqn):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigTaskAllocationInitParam.__init__(self)
        ConfigDqn.__init__(self)

        self.MODEL_TYPE = Q_MODEL.QModel.value


class ConfigTaskAllocationDoubleDqn(ConfigBase, ConfigTaskAllocationInitParam, ConfigDoubleDqn):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigTaskAllocationInitParam.__init__(self)
        ConfigDoubleDqn.__init__(self)

        self.MODEL_TYPE = Q_MODEL.QModel.value