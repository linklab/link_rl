from a_configuration.base.parameter_base_comparison import ParameterComparisonBase
from a_configuration.parameters.open_ai_gym.parameter_cartpole import ParameterCartPoleDqn, ParameterCartPoleReinforce, \
    ParameterCartPoleA2c


class ParameterComparisonCartPoleDqn(ParameterComparisonBase):
    def __init__(self):
        ParameterComparisonBase.__init__(self)

        self.ENV_NAME = "CartPole-v1"

        self.AGENT_PARAMETERS = [
            ParameterCartPoleDqn(),
            ParameterCartPoleDqn(),
            ParameterCartPoleDqn()
        ]

        for agent_parameter in self.AGENT_PARAMETERS:
            del agent_parameter.MAX_TRAINING_STEPS
            del agent_parameter.N_ACTORS
            del agent_parameter.N_EPISODES_FOR_MEAN_CALCULATION
            del agent_parameter.N_TEST_EPISODES
            del agent_parameter.N_VECTORIZED_ENVS
            del agent_parameter.PROJECT_HOME
            del agent_parameter.TEST_INTERVAL_TRAINING_STEPS
            del agent_parameter.TRAIN_INTERVAL_GLOBAL_TIME_STEPS
            del agent_parameter.USE_WANDB
            del agent_parameter.WANDB_ENTITY



# OnPolicy
class ParameterComparisonCartPoleReinforce(ParameterComparisonBase):
    def __init__(self):
        ParameterComparisonBase.__init__(self)

        self.AGENT_PARAMETERS = [
            ParameterCartPoleReinforce(),
            ParameterCartPoleReinforce(),
            ParameterCartPoleReinforce()
        ]


class ParameterComparisonCartPoleA2c(ParameterComparisonBase):
    def __init__(self):
        ParameterComparisonBase.__init__(self)

        self.AGENT_PARAMETERS = [
            ParameterCartPoleA2c(),
            ParameterCartPoleA2c(),
            ParameterCartPoleA2c(),
        ]
