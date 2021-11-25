from a_configuration.parameters.parameter_preamble import *


class ParameterComparison:
    USE_WANDB = True
    WANDB_ENTITY = "link-koreatech"
    AGENTS = [
        AgentType.Dqn,
        AgentType.A2c
    ]
    PARAMS_AGENTS = [
        ParameterCartPoleDqn,
        ParameterCartPoleA2c
    ]
    MAX_TRAINING_STEPS = 100_000
