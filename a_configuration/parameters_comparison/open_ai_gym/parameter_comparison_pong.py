from a_configuration.base.b_agents.agents_off_policy import ParameterDqn
from a_configuration.base.a_environments.open_ai_gym.gym_atari import ParameterPong
from a_configuration.base.c_models.convolutional_layers import ParameterMediumConvolutionalLayer
from a_configuration.base.parameter_base import ParameterBase
from a_configuration.base.parameter_base_comparison import ParameterComparisonBase
from a_configuration.parameters.open_ai_gym.parameter_pong import ParameterPongDqn


class ParameterComparisonPongDqn(ParameterComparisonBase):
    def __init__(self):
        ParameterComparisonBase.__init__(self)

        self.ENV_NAME = "PongNoFrameskip-v4"

        self.AGENT_PARAMETERS = [
            ParameterPongDqn(),
            ParameterPongDqn(),
            ParameterPongDqn()
        ]