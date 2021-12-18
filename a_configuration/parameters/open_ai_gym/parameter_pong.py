from a_configuration.base.b_agents.agents_off_policy import ParameterDqn
from a_configuration.base.a_environments.open_ai_gym.gym_atari import ParameterPong
from a_configuration.base.b_agents.agents_on_policy import ParameterA2c
from a_configuration.base.c_models.convolutional_layers import ParameterMediumConvolutionalLayer
from a_configuration.base.parameter_base import ParameterBase


class ParameterPongDqn(
    ParameterBase, ParameterPong, ParameterDqn, ParameterMediumConvolutionalLayer
):
    def __init__(self):
        ParameterBase.__init__(self)
        ParameterPong.__init__(self)
        ParameterDqn.__init__(self)
        ParameterMediumConvolutionalLayer.__init__(self)

        self.N_VECTORIZED_ENVS = 1
        self.N_ACTORS = 1
        self.MAX_TRAINING_STEPS = 1_000_000
        self.CONSOLE_LOG_INTERVAL_TRAINING_STEPS = 100
        self.TEST_INTERVAL_TRAINING_STEPS = 1024
        self.BUFFER_CAPACITY = 500_000


class ParameterPongA2c(
    ParameterBase, ParameterPong, ParameterA2c, ParameterMediumConvolutionalLayer
):
    def __init__(self):
        ParameterBase.__init__(self)
        ParameterPong.__init__(self)
        ParameterA2c.__init__(self)
        ParameterMediumConvolutionalLayer.__init__(self)

        self.N_VECTORIZED_ENVS = 1
        self.N_ACTORS = 1
        self.MAX_TRAINING_STEPS = 1_000_000
        self.CONSOLE_LOG_INTERVAL_TRAINING_STEPS = 1
        self.TEST_INTERVAL_TRAINING_STEPS = 1024
        self.BUFFER_CAPACITY = self.BATCH_SIZE
