from link_rl.a_configuration.a_base_config.a_environments.evolution_gym.config_evolution_gym_climber import \
    ConfigEvolutionGymClimberV0, ConfigEvolutionGymClimberV1, ConfigEvolutionGymClimberV2
from link_rl.a_configuration.a_base_config.b_agents.config_agents_off_policy import ConfigSac, ConfigTdmpc
from link_rl.a_configuration.a_base_config.b_agents.config_agents_on_policy import ConfigPpo
from link_rl.a_configuration.a_base_config.config_single_base import ConfigBase
from link_rl.d_models.d_basic_actor_critic_model import BASIC_ACTOR_CRITIC_MODEL
from link_rl.d_models.g_sac_model import SAC_MODEL
from link_rl.d_models.h_tdmpc_model import TDMPC_MODEL


class ConfigEvolutionGymClimberV0Sac(ConfigBase, ConfigEvolutionGymClimberV0, ConfigSac):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigEvolutionGymClimberV0.__init__(self)
        ConfigSac.__init__(self)

        self.MAX_TRAINING_STEPS = 1_000_000
        self.BUFFER_CAPACITY = 500_000
        self.ALPHA_LEARNING_RATE = 0.000025
        self.MIN_ALPHA = 0.2
        self.MODEL_TYPE = SAC_MODEL.ContinuousSacModel.value


class ConfigEvolutionGymClimberV1Sac(ConfigBase, ConfigEvolutionGymClimberV1, ConfigSac):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigEvolutionGymClimberV1.__init__(self)
        ConfigSac.__init__(self)

        self.MAX_TRAINING_STEPS = 1_000_000
        self.BUFFER_CAPACITY = 500_000
        self.ALPHA_LEARNING_RATE = 0.000025
        self.MIN_ALPHA = 0.2
        self.MODEL_TYPE = SAC_MODEL.ContinuousSacModel.value


class ConfigEvolutionGymClimberV2Sac(ConfigBase, ConfigEvolutionGymClimberV2, ConfigSac):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigEvolutionGymClimberV2.__init__(self)
        ConfigSac.__init__(self)

        self.MAX_TRAINING_STEPS = 1_000_000
        self.BUFFER_CAPACITY = 500_000
        self.ALPHA_LEARNING_RATE = 0.000025
        self.MIN_ALPHA = 0.2
        self.MODEL_TYPE = SAC_MODEL.ContinuousSacModel.value


class ConfigEvolutionGymClimberV0Ppo(ConfigBase, ConfigEvolutionGymClimberV0, ConfigPpo):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigEvolutionGymClimberV0.__init__(self)
        ConfigPpo.__init__(self)

        self.MAX_TRAINING_STEPS = 1_000_000
        self.BUFFER_CAPACITY = 500_000
        self.MODEL_TYPE = BASIC_ACTOR_CRITIC_MODEL.ContinuousBasicActorCriticSharedModel.value


class ConfigEvolutionGymClimberV1Ppo(ConfigBase, ConfigEvolutionGymClimberV1, ConfigPpo):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigEvolutionGymClimberV1.__init__(self)
        ConfigPpo.__init__(self)

        self.MAX_TRAINING_STEPS = 1_000_000
        self.BUFFER_CAPACITY = 500_000
        self.MODEL_TYPE = BASIC_ACTOR_CRITIC_MODEL.ContinuousBasicActorCriticSharedModel.value


class ConfigEvolutionGymClimberV2Ppo(ConfigBase, ConfigEvolutionGymClimberV2, ConfigPpo):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigEvolutionGymClimberV2.__init__(self)
        ConfigPpo.__init__(self)

        self.MAX_TRAINING_STEPS = 1_000_000
        self.BUFFER_CAPACITY = 500_000
        self.MODEL_TYPE = BASIC_ACTOR_CRITIC_MODEL.ContinuousBasicActorCriticSharedModel.value


class ConfigEvolutionGymClimberV0Tdmpc(ConfigBase, ConfigEvolutionGymClimberV0, ConfigTdmpc):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigEvolutionGymClimberV0.__init__(self)
        ConfigTdmpc.__init__(self)

        self.MAX_TRAINING_STEPS = 1_000_000
        self.MODEL_TYPE = TDMPC_MODEL.TdmpcModel.value
        self.FIXED_TOTAL_TIME_STEPS_PER_EPISODE = 400
        self.ACTION_REPEAT = 1

class ConfigEvolutionGymClimberV1Tdmpc(ConfigBase, ConfigEvolutionGymClimberV1, ConfigTdmpc):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigEvolutionGymClimberV1.__init__(self)
        ConfigTdmpc.__init__(self)

        self.MAX_TRAINING_STEPS = 1_000_000
        self.MODEL_TYPE = TDMPC_MODEL.TdmpcModel.value
        self.FIXED_TOTAL_TIME_STEPS_PER_EPISODE = 600
        self.ACTION_REPEAT = 1

class ConfigEvolutionGymClimberV2Tdmpc(ConfigBase, ConfigEvolutionGymClimberV2, ConfigTdmpc):
    def __init__(self):
        ConfigBase.__init__(self)
        ConfigEvolutionGymClimberV2.__init__(self)
        ConfigTdmpc.__init__(self)

        self.MAX_TRAINING_STEPS = 1_000_000
        self.MODEL_TYPE = TDMPC_MODEL.TdmpcModel.value
        self.FIXED_TOTAL_TIME_STEPS_PER_EPISODE = 1000
        self.ACTION_REPEAT = 1

