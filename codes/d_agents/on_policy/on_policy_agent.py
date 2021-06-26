import math
from abc import abstractmethod
import numpy as np
import torch

from codes.d_agents.a0_base_agent import BaseAgent
from codes.e_utils.names import AgentMode


class OnPolicyAgent(BaseAgent):
    """
    Abstract Agent interface
    """
    def __init__(self, worker_id, params, action_shape, action_min, action_max, device):
        super(OnPolicyAgent, self).__init__(worker_id, params, action_shape, action_min, action_max, device)
        self.model_version = 0
        self.buffer = None
        self.model = None
        self.train_action_selector = None

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

    def train_on_policy(self, step_idx, current_model_version):
        train_results = self.on_train(step_idx=step_idx, expected_model_version=current_model_version.value)
        current_model_version.value += 1
        return train_results

    @abstractmethod
    def on_train(self, step_idx, expected_model_version):
        raise NotImplementedError

    def discrete_call(self, states, agent_states):
        states = self.preprocess(states)

        if len(states) == 1:
            self.model.eval()
        else:
            self.model.train()

        if self.agent_mode == AgentMode.TRAIN:
            probs_v, value_v = self.model.base.forward(states)
            actions = self.train_action_selector(probs_v)

            return actions, agent_states
        else:
            probs_v = self.model.base.forward_actor(states)
            actions = self.test_and_play_action_selector(probs_v)
            return actions, None

    def continuous_stochastic_call(self, states, agent_states):
        states = self.preprocess(states)

        if len(states) == 1:
            self.model.eval()
        else:
            self.model.train()

        if self.agent_mode == AgentMode.TRAIN:
            mu_v, logstd_v, value_v = self.model.base(states)
            actions = self.train_action_selector(mu_v=mu_v, logstd_v=logstd_v)

            return actions, agent_states
        else:
            mu_v, _ = self.test_model.base.actor(states)
            actions = self.test_and_play_action_selector(mu_v, logstd_v=None)

            return actions, None

    # https://proofwiki.org/wiki/Differential_Entropy_of_Gaussian_Distribution
    def calc_entropy(self, logstd_v):
        return torch.log(logstd_v * math.sqrt(2 * np.pi)) + 0.5

    # def calc_logprob(self, mu_v, logstd_v, actions_v):
    #     p1 = -1.0 * ((mu_v - actions_v) ** 2) / (2 * torch.exp(logstd_v).clamp(min=1e-3, max=1e3) ** 2)
    #     p2 = -1.0 * torch.log(torch.sqrt(2 * np.pi * torch.exp(logstd_v).clamp(min=1e-3, max=1e3) ** 2))
    #     return p1 + p2
