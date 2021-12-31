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
    DQN = 0
    DOUBLE_DQN = 1
    DUELING_DQN = 2
    DOUBLE_DUELING_DQN = 3
    REINFORCE = 4
    A2C = 5
    DDPG = 6
    TD3 = 7
    SAC = 8
    PPO = 9


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