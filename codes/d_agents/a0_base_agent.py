import copy

import numpy as np
import torch


def float32_preprocessor(values):
    np_values = np.array(values, dtype=np.float32)
    return torch.tensor(np_values)

def long64_preprocessor(values):
    np_values = np.array(values, dtype=np.int64)
    return torch.tensor(np_values)


class BaseAgent:
    """
    Abstract Agent interface
    """
    def __init__(self):
        self.buffer = None
        pass

    def initial_agent_state(self):
        return np.array([0.0])

    def set_experience_source_to_buffer(self, experience_source):
        self.buffer.set_experience_source(experience_source)

    def __call__(self, states, agent_states):
        """
        Convert observations and states into actions to take
        :param states: list of environment states to process
        :param agent_states: list of states with the same length as observations
        :return: tuple of actions, states
        """
        assert isinstance(states, list)
        assert isinstance(agent_states, list)
        assert len(agent_states) == len(states)

        raise NotImplementedError


class TargetNet:
    """
    Wrapper around model which provides copy of it instead of trained weights
    """
    def __init__(self, model):
        self.model = model
        self.target_model = copy.deepcopy(model)

    def sync(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def alpha_sync(self, alpha):
        """
        Blend params of target net with params from the model
        :param alpha:
        """
        assert isinstance(alpha, float)
        assert 0.0 < alpha <= 1.0
        state = self.model.state_dict()
        tgt_state = self.target_model.state_dict()
        for k, v in state.items():
            tgt_state[k] = tgt_state[k] * alpha + (1.0 - alpha) * v
        self.target_model.load_state_dict(tgt_state)
