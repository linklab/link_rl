from a_configuration.a_base_config.a_environments.task_allocation.config_task_allocation import \
    ConfigTaskAllocationInitParam
from a_configuration.a_base_config.b_agents.config_agents_off_policy import ConfigDqn
from a_configuration.a_base_config.config_single_base import ConfigBase


class ConfigTaskAllocationDqn(ConfigBase, ConfigTaskAllocationInitParam, ConfigDqn):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigTaskAllocationInitParam.__init__(self)
        ConfigDqn.__init__(self)

