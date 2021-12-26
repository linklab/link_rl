from a_configuration.b_base.parameter_base_comparison import ParameterComparisonBase
from a_configuration.c_parameters.open_ai_gym.parameter_pong import ParameterPongDqn


class ParameterComparisonPongDqn(ParameterComparisonBase):
    def __init__(self):
        ParameterComparisonBase.__init__(self)
        self.ENV_NAME = "PongNoFrameskip-v4"

        self.TEST_INTERVAL_TRAINING_STEPS = 1_024
        self.MAX_TRAINING_STEPS = 300_000

        self.AGENT_PARAMETERS = [
            ParameterPongDqn(),
            ParameterPongDqn(),
            ParameterPongDqn()
        ]