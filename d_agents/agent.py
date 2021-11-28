from abc import abstractmethod
import torch.multiprocessing as mp

from g_utils.types import AgentMode, AgentType, OnPolicyAgentTypes


class Agent:
    def __init__(self, n_features, n_actions, device, params):
        self.n_features = n_features
        self.n_actions = n_actions
        self.device = device
        self.params = params

        self.model = None
        self.models_dir = None

    @abstractmethod
    def get_action(self, obs, mode=AgentMode.TRAIN):
        pass

    def train(self, buffer, total_time_steps_v=None, training_steps=None):
        loss = 0.0
        if self.params.AGENT_TYPE == AgentType.Dqn:
            if len(buffer) > self.params.MIN_BUFFER_SIZE_FOR_TRAIN:
                loss = self.train_dqn(
                    buffer=buffer, training_steps=training_steps
                )
                training_steps.value += 1
        elif self.params.AGENT_TYPE == AgentType.A2c:
            if len(buffer) > self.params.BATCH_SIZE:
                loss = self.train_a2c(buffer=buffer)
                training_steps.value += 1
        elif self.params.AGENT_TYPE == AgentType.Reinforce:
            if len(buffer) > 0:
                loss = self.train_reinforce(buffer=buffer)
                training_steps.value += 1

        # NOTE !!!
        if self.params.AGENT_TYPE in OnPolicyAgentTypes:
            buffer.clear()

        return loss

    @abstractmethod
    def train_dqn(self, buffer, training_steps):
        return 0.0

    @abstractmethod
    def train_reinforce(self, buffer):
        return 0.0

    @abstractmethod
    def train_a2c(self, buffer):
        return 0.0
