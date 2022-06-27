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

    TINY_1D_CONVOLUTIONAL = 5
    SMALL_1D_CONVOLUTIONAL = 6
    MEDIUM_1D_CONVOLUTIONAL = 7
    LARGE_1D_CONVOLUTIONAL = 8
    
    TINY_2D_CONVOLUTIONAL = 9
    SMALL_2D_CONVOLUTIONAL = 10
    MEDIUM_2D_CONVOLUTIONAL = 11
    LARGE_2D_CONVOLUTIONAL = 12

    TINY_RECURRENT = 13
    SMALL_RECURRENT = 14
    MEDIUM_RECURRENT = 15
    LARGE_RECURRENT = 16

    TINY_RECURRENT_1D_CONVOLUTIONAL = 17
    SMALL_RECURRENT_1D_CONVOLUTIONAL = 18
    MEDIUM_RECURRENT_1D_CONVOLUTIONAL = 19
    LARGE_RECURRENT_1D_CONVOLUTIONAL = 20

    TINY_RECURRENT_2D_CONVOLUTIONAL = 21
    SMALL_RECURRENT_2D_CONVOLUTIONAL = 22
    MEDIUM_RECURRENT_2D_CONVOLUTIONAL = 23
    LARGE_RECURRENT_2D_CONVOLUTIONAL = 24

    KNAPSACK_TEST = 25

class LayerActivationType(enum.Enum):
    LEAKY_RELU = 0
    ELU = 1
    PReLU = 2
    SELU = 3
    LINEAR = 4


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
    ASYNCHRONOUS_PPO = 8
    PPO_TRAJECTORY = 9
    DDPG = 10
    TD3 = 11
    SAC = 12
    TDMPC = 13
    Td3Drq2 = 14

class ConvolutionType(enum.Enum):
    ONE_DIMENSION = 0
    TWO_DIMENSION = 1
    THREE_DIMENSION = 2


class HerConstant:
    ACHIEVED_GOAL = "ACHIEVED_GOAL"
    DESIRED_GOAL = "DESIRED_GOAL"
    HER_SAVE_DONE = "HER_SAVE_DONE"   # HER_SAVE_DONE 이 True일 때만 HER_BUFFER에 SAVE


OnPolicyAgentTypes = [
    AgentType.REINFORCE, AgentType.A2C, AgentType.A3C, AgentType.PPO, AgentType.ASYNCHRONOUS_PPO, AgentType.PPO_TRAJECTORY
]

OffPolicyAgentTypes = [
    AgentType.DQN, AgentType.DOUBLE_DQN, AgentType.DUELING_DQN, AgentType.DOUBLE_DUELING_DQN,
    AgentType.DDPG, AgentType.TD3, AgentType.SAC, AgentType.TDMPC, AgentType.Td3Drq2
]

ActorCriticAgentTypes = [
    AgentType.A2C, AgentType.A3C, AgentType.PPO, AgentType.ASYNCHRONOUS_PPO, AgentType.PPO_TRAJECTORY, AgentType.DDPG, AgentType.TD3, AgentType.SAC
]
