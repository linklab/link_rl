from a_configuration.b_base.b_agents.agents_off_policy import ParameterDqn
from a_configuration.b_base.a_environments.open_ai_gym.gym_atari import ParameterPong
from a_configuration.b_base.b_agents.agents_on_policy import ParameterA2c
from a_configuration.b_base.c_models.convolutional_models import ParameterConvolutionalModel
from a_configuration.b_base.parameter_base import ParameterBase
from g_utils.types import ModelType


class ParameterPongDqn(
    ParameterBase, ParameterPong, ParameterDqn
):
    def __init__(self):
        ParameterBase.__init__(self)
        ParameterPong.__init__(self)
        ParameterDqn.__init__(self)

        self.N_VECTORIZED_ENVS = 1
        self.N_ACTORS = 1
        self.MAX_TRAINING_STEPS = 1_000_000
        self.CONSOLE_LOG_INTERVAL_TRAINING_STEPS = 100
        self.BUFFER_CAPACITY = 500_000
        self.MODEL = ParameterConvolutionalModel(ModelType.MEDIUM_CONVOLUTIONAL)


class ParameterPongA2c(
    ParameterBase, ParameterPong, ParameterA2c
):
    def __init__(self):
        ParameterBase.__init__(self)
        ParameterPong.__init__(self)
        ParameterA2c.__init__(self)

        self.BATCH_SIZE = 128
        self.LEARNING_RATE = 0.00001
        self.N_STEP = 1
        self.N_VECTORIZED_ENVS = 2
        self.N_ACTORS = 1
        self.MAX_TRAINING_STEPS = 1_000_000
        self.CONSOLE_LOG_INTERVAL_TRAINING_STEPS = 100
        self.TEST_INTERVAL_TRAINING_STEPS = 1_024
        self.ENTROPY_BETA = 0.0001
        self.BUFFER_CAPACITY = self.BATCH_SIZE
        self.MODEL = ParameterConvolutionalModel(ModelType.MEDIUM_CONVOLUTIONAL)

