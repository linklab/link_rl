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
    TINY_LINEAR = 0
    SMALL_LINEAR = 1
    SMALL_LINEAR_2 = 2
    MEDIUM_LINEAR = 3
    LARGE_LINEAR = 4

    SMALL_CONVOLUTIONAL = 5
    MEDIUM_CONVOLUTIONAL = 6
    LARGE_CONVOLUTIONAL = 7

    SMALL_RECURRENT = 8
    MEDIUM_RECURRENT = 9
    LARGE_RECURRENT = 10

    SMALL_RECURRENT_CONVOLUTIONAL = 11
    MEDIUM_RECURRENT_CONVOLUTIONAL = 12
    LARGE_RECURRENT_CONVOLUTIONAL = 13


class LayerActivationType(enum.Enum):
    LEAKY_RELU = 0
    ELU = 1


class LossFunctionType(enum.Enum):
    MSE_LOSS = 0
    HUBER_LOSS = 1


class AgentType(enum.Enum):
    DQN = 0
    DOUBLE_DQN = 1
    DUELING_DQN = 2
    DOUBLE_DUELING_DQN = 3
    REINFORCE = 4
    A2C = 5
    PPO = 6
    PPO_TRAJECTORY = 7
    DDPG = 8
    TD3 = 9
    SAC = 10


OnPolicyAgentTypes = [
    AgentType.REINFORCE, AgentType.A2C, AgentType.PPO
]

OffPolicyAgentTypes = [
    AgentType.DQN, AgentType.DOUBLE_DQN, AgentType.DUELING_DQN, AgentType.DOUBLE_DUELING_DQN,
    AgentType.DDPG, AgentType.TD3, AgentType.SAC
]

ActorCriticAgentTypes = [
    AgentType.A2C, AgentType.PPO, AgentType.DDPG, AgentType.TD3, AgentType.SAC
]
