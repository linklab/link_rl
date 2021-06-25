import math
from abc import abstractmethod
import numpy as np
import torch

from codes.a_config._rl_parameters.on_policy.parameter_on_policy import OnPolicyActionType
from codes.d_agents.a0_base_agent import BaseAgent, float32_preprocessor, long64_preprocessor
from codes.d_agents.actions import ActionStdTracker
from codes.e_utils.names import RLAlgorithmName, AgentMode


class OnPolicyAgent(BaseAgent):
    """
    Abstract Agent interface
    """
    def __init__(self, worker_id, params, action_shape, device):
        super(OnPolicyAgent, self).__init__(worker_id, params, action_shape, device)
        self.model_version = 0
        self.buffer = None
        self.model = None
        self.train_action_selector = None
        self.action_std_tracker = None

    def set_action_std_tracker(self):
        if self.params.TYPE_OF_ON_POLICY_ACTION == OnPolicyActionType.ACTION_STD_DECAY:
            self.action_std_tracker = ActionStdTracker(
                actor_model=self.model.base.actor,
                action_std_start=self.params.ACTION_STD_INIT,
                action_std_final=self.params.ACTION_STD_MIN,
                action_std_last_frames=self.params.ACTION_STD_MIN_STEP
            )

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

    # https://proofwiki.org/wiki/Differential_Entropy_of_Gaussian_Distribution
    def calc_entropy(self, logstd_v):
        return torch.log(logstd_v * math.sqrt(2 * np.pi)) + 0.5

    # def calc_logprob(self, mu_v, logstd_v, actions_v):
    #     p1 = -1.0 * ((mu_v - actions_v) ** 2) / (2 * torch.exp(logstd_v).clamp(min=1e-3, max=1e3) ** 2)
    #     p2 = -1.0 * torch.log(torch.sqrt(2 * np.pi * torch.exp(logstd_v).clamp(min=1e-3, max=1e3) ** 2))
    #     return p1 + p2
