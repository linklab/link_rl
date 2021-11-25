from g_utils.commons import AgentType


class ParameterReinforce:
    AGENT_TYPE = AgentType.Reinforce

    LEARNING_RATE = 0.0001
    BUFFER_CAPACITY = 1_000
    GAMMA = 0.99


class ParameterA2c:
    AGENT_TYPE = AgentType.A2c

    LEARNING_RATE = 0.0001
    BUFFER_CAPACITY = 1_000
    GAMMA = 0.99
    BATCH_SIZE = 64
    MIN_BUFFER_SIZE_FOR_TRAIN = BATCH_SIZE
