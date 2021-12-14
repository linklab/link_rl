from abc import abstractmethod
import torch.multiprocessing as mp

from g_utils.types import AgentMode, AgentType, OnPolicyAgentTypes


class Agent:
    def __init__(self, obs_shape, n_actions, device, parameter):
        self.obs_shape = obs_shape
        self.n_actions = n_actions
        self.device = device
        self.parameter = parameter

        self.model = None
        self.models_dir = None

    @abstractmethod
    def get_action(self, obs, mode=AgentMode.TRAIN):
        pass

    def train(self, buffer, training_steps_v=None):
        is_train_done = False
        if self.parameter.AGENT_TYPE == AgentType.Dqn:
            if len(buffer) > self.parameter.MIN_BUFFER_SIZE_FOR_TRAIN:
                self.train_dqn(
                    buffer=buffer, training_steps_v=training_steps_v
                )
                is_train_done = True
        elif self.parameter.AGENT_TYPE == AgentType.A2c:
            if len(buffer) > self.parameter.BATCH_SIZE:
                self.train_a2c(buffer=buffer)
                is_train_done = True
        elif self.parameter.AGENT_TYPE == AgentType.Reinforce:
            if len(buffer) > 0:
                self.train_reinforce(buffer=buffer)
                is_train_done = True

        # NOTE !!!
        if is_train_done:
            if self.parameter.AGENT_TYPE in OnPolicyAgentTypes:
                buffer.clear()

        return is_train_done

    @abstractmethod
    def train_dqn(self, buffer, training_steps_v):
        return 0.0

    @abstractmethod
    def train_reinforce(self, buffer):
        return 0.0

    @abstractmethod
    def train_a2c(self, buffer):
        return 0.0
