from collections import namedtuple
import enum

Transition = namedtuple(
    typename='Transition',
    field_names=[
        'observation', 'action', 'next_observation', 'reward', 'done', 'info'
    ],
    defaults=[None] * 6
)

Episode_history = namedtuple(
    typename='Episode_history',
    field_names=[
        'observation_history', 'action_history', 'reward_history', 'to_play_history',
        'child_visits_history', 'root_values_history', 'info_history', 'done'
    ],
    defaults=[None] * 8
)

# Transitions = namedtuple(
#     typename='Transitions',
#     field_names=[
#         'observations', 'actions', 'next_observations', 'rewards', 'dones', 'infos'
#     ],
#     defaults=[None] * 6
# )


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

    TINY_CONVOLUTIONAL = 5
    SMALL_CONVOLUTIONAL = 6
    MEDIUM_CONVOLUTIONAL = 7
    LARGE_CONVOLUTIONAL = 8

    SMALL_RECURRENT = 9
    MEDIUM_RECURRENT = 10
    LARGE_RECURRENT = 11

    SMALL_RECURRENT_CONVOLUTIONAL = 12
    MEDIUM_RECURRENT_CONVOLUTIONAL = 13
    LARGE_RECURRENT_CONVOLUTIONAL = 14


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
    A3C = 6
    PPO = 7
    PPO_TRAJECTORY = 8
    DDPG = 9
    TD3 = 10
    SAC = 11
    MUZERO = 12


OnPolicyAgentTypes = [
    AgentType.REINFORCE, AgentType.A2C, AgentType.PPO, AgentType.PPO_TRAJECTORY
]

OffPolicyAgentTypes = [
    AgentType.DQN, AgentType.DOUBLE_DQN, AgentType.DUELING_DQN, AgentType.DOUBLE_DUELING_DQN,
    AgentType.DDPG, AgentType.TD3, AgentType.SAC, AgentType.MUZERO
]

ActorCriticAgentTypes = [
    AgentType.A2C, AgentType.A3C, AgentType.PPO, AgentType.PPO_TRAJECTORY, AgentType.DDPG, AgentType.TD3, AgentType.SAC
]
