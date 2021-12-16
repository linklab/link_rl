from a_configuration.parameter_preamble import *


class ParameterComparison(ParameterComparisonCartPoleDqn):
    def __init__(self):
        super(ParameterComparison, self).__init__()
        self.USE_WANDB = True
        self.WANDB_ENTITY = "link-koreatech"

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
            del agent_parameter.TRAIN_INTERVAL_TOTAL_TIME_STEPS
            del agent_parameter.USE_WANDB
            del agent_parameter.WANDB_ENTITY

        self.AGENT_PARAMETERS[0].N_STEP = 1
        self.AGENT_PARAMETERS[1].N_STEP = 2
        self.AGENT_PARAMETERS[2].N_STEP = 3

        self.AGENT_LABELS = [
            "DQN (N_STEP=1)",
            "DQN (N_STEP=2)",
            "DQN (N_STEP=3)",
        ]

        self.MAX_TRAINING_STEPS = 1_000
