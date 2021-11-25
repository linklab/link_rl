from abc import abstractmethod
import torch.multiprocessing as mp

from g_utils.types import AgentMode, AgentType


class Agent:
    def __init__(self, n_features, n_actions, device, params):
        self.n_features = n_features
        self.n_actions = n_actions
        self.device = device
        self.params = params

        self.model = None
        self.models_dir = None
        self.model_version = mp.Value('i', 0)

    @abstractmethod
    def get_action(self, obs, mode=AgentMode.TRAIN):
        pass

    def train(self, buffer, total_time_steps_v, training_steps):
        loss = 0.0
        if self.params.AGENT_TYPE == AgentType.Dqn:
            if len(buffer) > self.params.MIN_BUFFER_SIZE_FOR_TRAIN:
                # TRAIN POLICY
                loss = self.train_dqn(
                    buffer=buffer,
                    total_time_steps_v=total_time_steps_v,
                    training_steps=training_steps
                )
                training_steps.value += 1
                self.model_version.value += 1
        elif self.params.AGENT_TYPE == AgentType.A2c:
            filtered_buffer = buffer.get_filtered_buffer(
                model_version_v=self.model_version.value
            )
            if len(filtered_buffer) > 0:
                loss = self.train_a2c(
                    filtered_buffer=filtered_buffer, device=self.device
                )
                training_steps.value += 1
                self.model_version.value += 1
        return loss

    def train_per_episode(self, buffer, training_steps):
        loss = None
        if self.params.AGENT_TYPE == AgentType.Reinforce:
            filtered_buffer = buffer.get_filtered_buffer(
                model_version_v=self.model_version.value
            )
            if len(filtered_buffer) > 0:
                loss = self.train_reinforce(
                    filtered_buffer=filtered_buffer, device=self.device
                )
                training_steps.value += 1
                self.model_version.value += 1
        return loss

    @abstractmethod
    def train_dqn(self, buffer, total_time_steps_v, training_steps):
        return 0.0

    @abstractmethod
    def train_reinforce(self, filtered_buffer, device):
        return 0.0

    @abstractmethod
    def train_a2c(self, filtered_buffer, device):
        return 0.0
