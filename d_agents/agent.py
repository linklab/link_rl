from abc import abstractmethod, ABC
import torch.multiprocessing as mp
from gym.spaces import Discrete, Box, MultiDiscrete

import numpy as np
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
        # Tuple
        # Discrete
        # MultiBinary
        # MultiDiscrete
        self.observation_shape = observation_space.shape
        self.action_shape = action_space.shape

        if isinstance(action_space, Discrete):
            self.n_out_actions = 1
            self.n_discrete_actions = action_space.n
        elif isinstance(action_space, MultiDiscrete):
            # TODO: Multi Discrete Space
            pass
        elif isinstance(action_space, Box):
            self.n_out_actions = action_space.shape[0]
            self.n_discrete_actions = None
        else:
            raise ValueError()

        self.buffer = Buffer(capacity=parameter.BUFFER_CAPACITY, device=self.device)

        self.model = None
        self.last_model_grad_max = mp.Value('d', 0.0)
        self.last_model_grad_l2 = mp.Value('d', 0.0)

    @abstractmethod
    def get_action(self, obs, mode=AgentMode.TRAIN):
        pass

    def before_train(self):
        pass

    def train(self, training_steps_v=None):
        self.before_train()

        is_train_success_done = False
        if self.parameter.AGENT_TYPE in [AgentType.DQN, AgentType.DUELING_DQN]:
            if len(self.buffer) >= self.parameter.MIN_BUFFER_SIZE_FOR_TRAIN:
                self.train_dqn(training_steps_v=training_steps_v)
                is_train_success_done = True
        elif self.parameter.AGENT_TYPE == AgentType.DOUBLE_DQN:
            if len(self.buffer) >= self.parameter.MIN_BUFFER_SIZE_FOR_TRAIN:
                self.train_ddqn(training_steps_v=training_steps_v)
                is_train_success_done = True
        elif self.parameter.AGENT_TYPE == AgentType.A2C:
            if len(self.buffer) >= self.parameter.BATCH_SIZE:
                self.train_a2c()
                is_train_success_done = True
        elif self.parameter.AGENT_TYPE == AgentType.DDPG:
            if len(self.buffer) >= self.parameter.BATCH_SIZE:
                self.train_ddpg()
                is_train_success_done = True
        elif self.parameter.AGENT_TYPE == AgentType.REINFORCE:
            if len(self.buffer) > 0:
                self.train_reinforce()
                is_train_success_done = True
        else:
            raise ValueError()

        # NOTE !!!
        if is_train_success_done:
            if self.parameter.AGENT_TYPE in OnPolicyAgentTypes:
                self.buffer.clear()

            self.after_train()

        return is_train_success_done

    def after_train(self):
        grads = np.concatenate(
            [p.grad.data.cpu().numpy().flatten() for p in self.model.parameters() if p.grad is not None]
        )
        self.last_model_grad_l2.value = np.sqrt(np.mean(np.square(grads)))
        self.last_model_grad_max.value = np.max(np.abs(grads))

    @abstractmethod
    def train_dqn(self, training_steps_v):
        return 0.0

    @abstractmethod
    def train_ddqn(self, training_steps_v):
        return 0.0

    @abstractmethod
    def train_reinforce(self):
        return 0.0

    @abstractmethod
    def train_a2c(self):
        return 0.0

    @abstractmethod
    def train_ddpg(self):
        return 0.0

    def synchronize_models(self, source_model, target_model):
        target_model.load_state_dict(source_model.state_dict())

    def soft_synchronize_models(self, source_model, target_model, tau):
        assert isinstance(tau, float)
        assert 0.0 < tau <= 1.0

        source_model_state = source_model.state_dict()
        target_model_state = target_model.state_dict()
        for k, v in source_model_state.items():
            target_model_state[k] = (1.0 - tau) * target_model_state[k] + tau * v
        target_model.load_state_dict(target_model_state)

# class DiscreteActionAgent(Agent, ABC):
#     def __init__(self, observation_shape, n_discrete_actions, device, parameter):
#         super(DiscreteActionAgent, self).__init__(observation_shape, device, parameter)
#         self.n_discrete_actions = n_discrete_actions
#
#
# class ContinuousActionAgent(Agent, ABC):
#     def __init__(self, observation_shape, action_shape, device, parameter):
#         super(ContinuousActionAgent, self).__init__(observation_shape, device, parameter)
#         self.action_shape = action_shape
#         self.n_out_actions = self.action_shape[0]
