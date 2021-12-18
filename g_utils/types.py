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
    LINEAR = 0
    CONVOLUTIONAL = 1
    RECURRENT = 2


class AgentType(enum.Enum):
    Dqn = 0
    Reinforce = 1
    A2c = 2
    Ddpg = 3


OnPolicyAgentTypes = [AgentType.Reinforce, AgentType.A2c]
OffPolicyAgentTypes = [AgentType.Dqn]
