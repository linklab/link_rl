from a_configuration.b_base.a_environments.open_ai_gym.gym_box2d import ParameterLunarLanderContinuous, \
    ParameterLunarLander
from a_configuration.b_base.b_agents.agents_off_policy import ParameterDqn, ParameterDdpg, ParameterSac
from a_configuration.b_base.b_agents.agents_on_policy import ParameterA2c, ParameterReinforce
from a_configuration.b_base.c_models.linear_models import ParameterLinearModel
from a_configuration.b_base.parameter_base import ParameterBase
from a_configuration.b_base.a_environments.open_ai_gym.gym_classic_control import ParameterCartPole
from g_utils.types import ModelType


class ParameterLunarLanderA2c(ParameterBase, ParameterLunarLander, ParameterA2c):
    def __init__(self):
        ParameterBase.__init__(self)
        ParameterLunarLander.__init__(self)
        ParameterA2c.__init__(self)

        self.N_VECTORIZED_ENVS = 1
        self.N_ACTORS = 1
        self.MAX_TRAINING_STEPS = 100_000
        self.BUFFER_CAPACITY = 100_000
        self.BATCH_SIZE = 64
        self.CONSOLE_LOG_INTERVAL_TRAINING_STEPS = 100
        self.MODEL = ParameterLinearModel(ModelType.MEDIUM_LINEAR)


class ParameterLunarLanderContinuousA2c(ParameterBase, ParameterLunarLanderContinuous, ParameterA2c):
    def __init__(self):
        ParameterBase.__init__(self)
        ParameterLunarLanderContinuous.__init__(self)
        ParameterA2c.__init__(self)

        self.N_VECTORIZED_ENVS = 1
        self.N_ACTORS = 1
        self.MAX_TRAINING_STEPS = 200_000
        self.BUFFER_CAPACITY = 200_000
        self.BATCH_SIZE = 64
        self.LEARNING_RATE = 0.00075
        self.CONSOLE_LOG_INTERVAL_TRAINING_STEPS = 100
        self.MODEL = ParameterLinearModel(ModelType.MEDIUM_LINEAR)


class ParameterLunarLanderContinuousDdpg(ParameterBase, ParameterLunarLanderContinuous, ParameterDdpg):
    def __init__(self):
        ParameterBase.__init__(self)
        ParameterLunarLanderContinuous.__init__(self)
        ParameterDdpg.__init__(self)

        self.N_VECTORIZED_ENVS = 1
        self.N_ACTORS = 1
        self.MAX_TRAINING_STEPS = 200_000
        self.BUFFER_CAPACITY = 200_000
        self.BATCH_SIZE = 64
        self.LEARNING_RATE = 0.0001
        self.CONSOLE_LOG_INTERVAL_TRAINING_STEPS = 100
        self.MODEL = ParameterLinearModel(ModelType.MEDIUM_LINEAR)


class ParameterLunarLanderContinuousSac(ParameterBase, ParameterLunarLanderContinuous, ParameterSac):
    def __init__(self):
        ParameterBase.__init__(self)
        ParameterLunarLanderContinuous.__init__(self)
        ParameterSac.__init__(self)

        self.N_VECTORIZED_ENVS = 1
        self.N_ACTORS = 1
        self.MAX_TRAINING_STEPS = 500_000
        self.BUFFER_CAPACITY = 200_000
        self.BATCH_SIZE = 64
        self.LEARNING_RATE = 0.0001
        self.CONSOLE_LOG_INTERVAL_TRAINING_STEPS = 100
        self.MODEL = ParameterLinearModel(ModelType.MEDIUM_LINEAR)
