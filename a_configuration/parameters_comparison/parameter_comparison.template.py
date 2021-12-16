from a_configuration.parameter_preamble import *


class ParameterComparison(ParameterComparisonCartPoleDqn):
    def __init__(self):
        super(ParameterComparison, self).__init__()
        self.USE_WANDB = True
        self.WANDB_ENTITY = "link-koreatech"

        self.AGENT_PARAMETERS[0].N_STEP = 1
        self.AGENT_PARAMETERS[1].N_STEP = 2
        self.AGENT_PARAMETERS[2].N_STEP = 3

        self.AGENT_LABELS = [
            "DQN (N_STEP=1)",
            "DQN (N_STEP=2)",
            "DQN (N_STEP=3)",
        ]


