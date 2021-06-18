import math
from abc import abstractmethod
import numpy as np
import torch

from codes.d_agents.a0_base_agent import BaseAgent, float32_preprocessor, long64_preprocessor
from codes.e_utils.names import RLAlgorithmName, AgentMode


class OnPolicyAgent(BaseAgent):
    """
    Abstract Agent interface
    """
    def __init__(self, worker_id, params, action_shape, device):
        super(OnPolicyAgent, self).__init__(worker_id, params, action_shape, device)

    @abstractmethod
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

    @abstractmethod
    def train(self, step_idx):
        raise NotImplementedError

    # https://proofwiki.org/wiki/Differential_Entropy_of_Gaussian_Distribution
    def calc_entropy(self, logstd_v):
        return torch.log(logstd_v * math.sqrt(2 * np.pi)) + 1.0 / 2.0

    # def calc_logprob(self, mu_v, logstd_v, actions_v):
    #     p1 = -1.0 * ((mu_v - actions_v) ** 2) / (2 * torch.exp(logstd_v).clamp(min=1e-3, max=1e3) ** 2)
    #     p2 = -1.0 * torch.log(torch.sqrt(2 * np.pi * torch.exp(logstd_v).clamp(min=1e-3, max=1e3) ** 2))
    #     return p1 + p2
