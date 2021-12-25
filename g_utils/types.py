from collections import namedtuple
import enum

Transition = namedtuple(
    typename='Transition',
    field_names=[
        'observation', 'action', 'next_observation',
        'reward', 'done', 'info'
    ],
    defaults=[None] * 6
)

Transitions = namedtuple(
    typename='Transitions',
    field_names=[
        'observations', 'actions', 'next_observations',
        'rewards', 'dones', 'infos'
    ],
    defaults=[None] * 6
)


class AgentMode(enum.Enum):
    TRAIN = 0
    TEST = 1
    PLAY = 2


class ModelType(enum.Enum):
    SMALL_LINEAR = 0
    MEDIUM_LINEAR = 1
    LARGE_LINEAR = 2

    SMALL_CONVOLUTIONAL = 3
    MEDIUM_CONVOLUTIONAL = 4
    LARGE_CONVOLUTIONAL = 5

    SMALL_RECURRENT = 6
    MEDIUM_RECURRENT = 7
    LARGE_RECURRENT = 8


class AgentType(enum.Enum):
    Dqn = 0
    DOUBLE_DQN = 1
    DUELING_DQN = 2
    DOUBLE_DUELING_DQN = 3
    Reinforce = 4
    A2c = 5
    Ddpg = 6


OnPolicyAgentTypes = [AgentType.Reinforce, AgentType.A2c]
OffPolicyAgentTypes = [AgentType.Dqn]
