from abc import abstractmethod
from collections import namedtuple

import numpy as np
import torch
from torch.distributions import Normal, Categorical

from codes.c_models.base_model import RNNModel
from codes.c_models.continuous_action.continuous_action_model import ContinuousActionModel
from codes.c_models.discrete_action.discrete_action_model import DiscreteActionModel
from codes.e_utils.common_utils import show_info, float32_preprocessor, long64_preprocessor
from codes.e_utils.names import AgentMode, RLAlgorithmName


class BaseAgent:
    """
    Abstract Agent interface
    """
    def __init__(self, worker_id, action_shape, params, device):
        self.worker_id = worker_id

        self.model = None
        self.test_model = None

        self.train_action_selector = None
        self.test_and_play_action_selector = None

        self.params = params
        self.action_shape = action_shape
        self.device = device
        self.buffer = None
        self.agent_mode = AgentMode.TRAIN

    def preprocess(self, state):
        if not isinstance(state, torch.FloatTensor):
            state = float32_preprocessor(state).to(self.device)

        return state

    def set_experience_source_to_buffer(self, experience_source):
        self.buffer.set_experience_source(experience_source)

    @abstractmethod
    def __call__(self, state, agent_state):
        """
        Convert observations and state into actions to take
        :param state: list of environment state to process
        :param agent_state: list of state with the same length as observations
        :return: tuple of actions, state
        """
        assert isinstance(state, list)
        assert isinstance(agent_state, list)
        assert len(agent_state) == len(state)

        raise NotImplementedError

    @abstractmethod
    def train_off_policy(self, step_idx):
        raise NotImplementedError

    @abstractmethod
    def train_on_policy(self, step_idx, expected_model_version):
        raise NotImplementedError

    def convert_action_to_torch_tensor(self, values, device):
        if isinstance(self.model, DiscreteActionModel):
            actions_v = long64_preprocessor(values).to(device)
        elif isinstance(self.model, ContinuousActionModel):
            actions_v = float32_preprocessor(values).to(device)
        else:
            raise ValueError()

        return actions_v
