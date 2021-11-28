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


class AgentType(enum.Enum):
    Dqn = 0
    Reinforce = 1
    A2c = 2


OnPolicyAgentTypes = [AgentType.Reinforce, AgentType.A2c]


OffPolicyAgentTypes = [AgentType.Dqn]


class AgentMode(enum.Enum):
    TRAIN = 0
    TEST = 1
    PLAY = 2


class EpsilonTracker:
    def __init__(self, epsilon_init, epsilon_final, epsilon_final_time_step_percent, max_training_steps):
        self.epsilon_init = epsilon_init
        self.epsilon_final = epsilon_final
        self.epsilon_final_time_step = max_training_steps * epsilon_final_time_step_percent

    def epsilon(self, training_step):
        epsilon = max(
            self.epsilon_init - training_step / self.epsilon_final_time_step,
            self.epsilon_final
        )
        return epsilon