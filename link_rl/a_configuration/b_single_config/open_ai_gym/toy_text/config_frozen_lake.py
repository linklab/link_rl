from link_rl.a_configuration.a_base_config.b_agents.config_agents_off_policy import ConfigDqn
from link_rl.a_configuration.a_base_config.config_single_base import ConfigBase
from link_rl.a_configuration.a_base_config.a_environments.open_ai_gym.config_gym_toy_text import ConfigFrozenLake
from link_rl.g_utils.types import ModelType


class ConfigFrozenLakeDqn(ConfigBase, ConfigFrozenLake, ConfigDqn):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigFrozenLake.__init__(self)
        ConfigDqn.__init__(self)

        self.MAX_TRAINING_STEPS = 100_000
        self.BUFFER_CAPACITY = 50_000

        if self.BOX_OBSERVATION:
            self.MODEL_TYPE = ModelType.TINY_2D_CONVOLUTIONAL
        else:
            self.MODEL_TYPE = ModelType.TINY_LINEAR


# class ConfigFrozenLakeDoubleDqn(ConfigBase, ConfigFrozenLake, ConfigDoubleDqn):
#     def __init__(self):
#         ConfigBase.__init__(self)
#         ConfigFrozenLake.__init__(self)
#         ConfigDoubleDqn.__init__(self)
#
#         self.N_VECTORIZED_ENVS = 1
#         self.N_ACTORS = 1
#         self.MAX_TRAINING_STEPS = 100_000
#         self.BUFFER_CAPACITY = 50_000
#         self.MODEL_TYPE = ModelType.SMALL_LINEAR
#
#
# class ConfigFrozenLakeDuelingDqn(ConfigBase, ConfigFrozenLake, ConfigDuelingDqn):
#     def __init__(self):
#         ConfigBase.__init__(self)
#         ConfigFrozenLake.__init__(self)
#         ConfigDuelingDqn.__init__(self)
#
#         self.N_VECTORIZED_ENVS = 1
#         self.N_ACTORS = 1
#         self.MAX_TRAINING_STEPS = 100_000
#         self.BUFFER_CAPACITY = 50_000
#         self.MODEL_TYPE = ModelType.SMALL_LINEAR
#
#
# class ConfigFrozenLakeDoubleDuelingDqn(ConfigBase, ConfigFrozenLake, ConfigDoubleDuelingDqn):
#     def __init__(self):
#         ConfigBase.__init__(self)
#         ConfigFrozenLake.__init__(self)
#         ConfigDoubleDuelingDqn.__init__(self)
#
#         self.N_VECTORIZED_ENVS = 1
#         self.N_ACTORS = 1
#         self.MAX_TRAINING_STEPS = 100_000
#         self.BUFFER_CAPACITY = 50_000
#         self.MODEL_TYPE = ModelType.SMALL_LINEAR
#
# # OnPolicy
#
#
# class ConfigFrozenLakeReinforce(ConfigBase, ConfigFrozenLake, ConfigReinforce):
#     def __init__(self):
#         ConfigBase.__init__(self)
#         ConfigFrozenLake.__init__(self)
#         ConfigReinforce.__init__(self)
#
#         self.MAX_TRAINING_STEPS = 100_000
#         self.MODEL_TYPE = ModelType.SMALL_LINEAR
#
#
# class ConfigFrozenLakeA2c(ConfigBase, ConfigFrozenLake, ConfigA2c):
#     def __init__(self):
#         ConfigBase.__init__(self)
#         ConfigFrozenLake.__init__(self)
#         ConfigA2c.__init__(self)
#
#         self.MAX_TRAINING_STEPS = 100_000
#         self.MODEL_TYPE = ModelType.SMALL_LINEAR
#
#
# class ConfigFrozenLakePpo(ConfigBase, ConfigFrozenLake, ConfigPpo):
#     def __init__(self):
#         ConfigBase.__init__(self)
#         ConfigFrozenLake.__init__(self)
#         ConfigPpo.__init__(self)
#
#         self.MAX_TRAINING_STEPS = 100_000
#         self.MODEL_TYPE = ModelType.SMALL_LINEAR
#
#
# class ConfigFrozenLakePpoTrajectory(ConfigBase, ConfigFrozenLake, ConfigPpoTrajectory):
#     def __init__(self):
#         ConfigBase.__init__(self)
#         ConfigFrozenLake.__init__(self)
#         ConfigPpoTrajectory.__init__(self)
#
#         self.MAX_TRAINING_STEPS = 100_000
#         self.MODEL_TYPE = ModelType.SMALL_LINEAR
