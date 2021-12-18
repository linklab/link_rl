from abc import abstractmethod, ABC
import torch.multiprocessing as mp
from gym.spaces import Discrete, Box

from g_utils.buffers import Buffer
from g_utils.types import AgentMode, AgentType, OnPolicyAgentTypes


class Agent:
    def __init__(self, observation_space, action_space, device, parameter):
        self.observation_space = observation_space
        self.action_space = action_space
        self.device = device
        self.parameter = parameter

        # Box
        # Dict
        # Discrete
        # MultiBinary
        # MultiDiscrete
        self.observation_shape = observation_space.shape
        self.action_shape = action_space.shape

        if isinstance(action_space, Discrete):
            self.n_out_actions = action_space.n
        elif isinstance(action_space, Box):
            self.n_out_actions = action_space.shape[0]
        else:
            raise ValueError()

        self.buffer = Buffer(capacity=parameter.BUFFER_CAPACITY, device=self.device)

        self.model = None
        self.models_dir = None

    def add_transition_to_buffer(self, n_step_transition):
        self.buffer.append(n_step_transition)

    @abstractmethod
    def get_action(self, obs, mode=AgentMode.TRAIN):
        pass

    def train(self, training_steps_v=None):
        is_train_success_done = False
        if self.parameter.AGENT_TYPE == AgentType.Dqn:
            if len(self.buffer) >= self.parameter.MIN_BUFFER_SIZE_FOR_TRAIN:
                self.train_dqn(training_steps_v=training_steps_v)
                is_train_success_done = True
        elif self.parameter.AGENT_TYPE == AgentType.A2c:
            if len(self.buffer) >= self.parameter.BATCH_SIZE:
                self.train_a2c()
                is_train_success_done = True
        elif self.parameter.AGENT_TYPE == AgentType.Reinforce:
            if len(self.buffer) > 0:
                self.train_reinforce()
                is_train_success_done = True

        # NOTE !!!
        if is_train_success_done:
            if self.parameter.AGENT_TYPE in OnPolicyAgentTypes:
                self.buffer.clear()

        return is_train_success_done

    @abstractmethod
    def train_dqn(self, training_steps_v):
        return 0.0

    @abstractmethod
    def train_reinforce(self):
        return 0.0

    @abstractmethod
    def train_a2c(self):
        return 0.0


class DiscreteActionAgent(Agent, ABC):
    def __init__(self, observation_shape, n_discrete_actions, device, parameter):
        super(DiscreteActionAgent, self).__init__(observation_shape, device, parameter)
        self.n_discrete_actions = n_discrete_actions


class ContinuousActionAgent(Agent, ABC):
    def __init__(self, observation_shape, action_shape, device, parameter):
        super(ContinuousActionAgent, self).__init__(observation_shape, device, parameter)
        self.action_shape = action_shape
        self.n_out_actions = self.action_shape[0]
