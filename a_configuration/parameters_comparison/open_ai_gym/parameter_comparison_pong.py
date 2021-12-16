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

        for agent_parameter in self.AGENT_PARAMETERS:
            del agent_parameter.MAX_TRAINING_STEPS
            del agent_parameter.N_ACTORS
            del agent_parameter.N_EPISODES_FOR_MEAN_CALCULATION
            del agent_parameter.N_TEST_EPISODES
            del agent_parameter.N_VECTORIZED_ENVS
            del agent_parameter.PROJECT_HOME
            del agent_parameter.TEST_INTERVAL_TRAINING_STEPS
            del agent_parameter.TRAIN_INTERVAL_TOTAL_TIME_STEPS
            del agent_parameter.USE_WANDB
            del agent_parameter.WANDB_ENTITY
