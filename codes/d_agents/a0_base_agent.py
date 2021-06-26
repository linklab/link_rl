import copy
import math
from abc import abstractmethod

import numpy as np
import torch

from codes.e_utils.names import AgentMode, RLAlgorithmName


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
    def __init__(self, worker_id, params, action_shape, action_min, action_max, device):
        self.worker_id = worker_id

        self.model = None
        self.test_model = None

        self.train_action_selector = None
        self.test_and_play_action_selector = None

        self.params = params
        self.action_shape = action_shape
        self.action_min = action_min
        self.action_max = action_max
        self.device = device
        self.buffer = None
        self.agent_mode = AgentMode.TRAIN
        pass

    def initial_agent_state(self):
        return np.zeros(shape=self.action_shape, dtype=np.float32)

    def preprocess(self, states):
        if not isinstance(states, torch.FloatTensor):
            states = float32_preprocessor(states).to(self.device)

        return states

    def set_experience_source_to_buffer(self, experience_source):
        self.buffer.set_experience_source(experience_source)

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
    def train_off_policy(self, step_idx):
        raise NotImplementedError

    @abstractmethod
    def train_on_policy(self, step_idx, expected_model_version):
        raise NotImplementedError

    def discrete_call(self, states, critics):
        states = self.preprocess(states)

        with torch.no_grad():
            probs_v = self.model.base.forward_actor(states)

        if self.agent_mode == AgentMode.TRAIN:
            actions = self.train_action_selector(probs_v)
        else:
            actions = self.test_and_play_action_selector(probs_v)

        critics = torch.zeros(size=probs_v.size())
        return actions, critics

    def continuous_stochastic_call(self, states):
        states = self.preprocess(states)

        if len(states) == 1:
            self.model.eval()
        else:
            self.model.train()

        if self.agent_mode == AgentMode.TRAIN:
            with torch.no_grad():
                mu_v = self.model.base.actor(states)
                actions = self.train_action_selector(
                    mu_v=mu_v,
                    action_variance=self.model.base.actor.action_variance
                )
        else:
            with torch.no_grad():
                mu_v = self.test_model.base.actor(states)
                actions = self.test_and_play_action_selector(mu_v, action_variance=None)

        critics = torch.zeros(size=mu_v.size())

        return actions, critics

    def continuous_sac_call(self, states):
        states = self.preprocess(states)

        if len(states) == 1:
            self.model.eval()
        else:
            self.model.train()

        if self.agent_mode == AgentMode.TRAIN:
            with torch.no_grad():
                mu_v, logstd_v = self.model.base.actor(states)
                actions = self.train_action_selector(mu_v=mu_v, logstd_v=logstd_v)
        else:
            with torch.no_grad():
                mu_v, _ = self.test_model.base.actor(states)
                actions = self.test_and_play_action_selector(mu_v=mu_v, logstd_v=None)

        critics = torch.zeros(size=mu_v.size())

        return actions, critics

    def unpack_batch_for_actor_critic(self, batch, model, params, sac=False, alpha=None):
        """
        Convert batch into training tensors
        :param batch:
        :param model:
        :return: states variable, actions tensor, target values variable
        """
        states, actions, rewards, not_done_idx, last_states = [], [], [], [], []

        for idx, exp in enumerate(batch):
            states.append(np.array(exp.state, copy=False))
            actions.append(exp.action)
            rewards.append(exp.reward)
            if exp.last_state is not None:
                not_done_idx.append(idx)
                last_states.append(np.array(exp.last_state, copy=False))

        states_v = float32_preprocessor(states).to(self.device)

        if params.RL_ALGORITHM in [RLAlgorithmName.DISCRETE_A2C_V0, RLAlgorithmName.DISCRETE_PPO_V0]:
            actions_v = long64_preprocessor(actions).to(self.device)
        elif params.RL_ALGORITHM in [RLAlgorithmName.CONTINUOUS_A2C_V0, RLAlgorithmName.CONTINUOUS_PPO_V0, RLAlgorithmName.SAC_V0]:
            actions_v = float32_preprocessor(actions).to(self.device)
        else:
            raise ValueError()

        # handle rewards
        target_action_values_np = np.array(rewards, dtype=np.float32)

        if not_done_idx:
            last_states_v = torch.FloatTensor(np.array(last_states, copy=False)).to(self.device)
            if sac:
                last_actions_v, last_entropies_v = model.sample(last_states_v)
                last_q_1_v, last_q_2_v = model.base.twinq(last_states_v, last_actions_v)
                last_q_v = torch.min(last_q_1_v, last_q_2_v) * (params.GAMMA ** params.N_STEP)
                last_q_v += alpha * last_entropies_v
                last_q_v = last_q_v.squeeze(-1)
                target_action_values_np[not_done_idx] += last_q_v.data.cpu().numpy()
            else:
                last_values_v = model.base.forward_critic(last_states_v)
                last_values_np = last_values_v.data.cpu().numpy()[:, 0] * (params.GAMMA ** params.N_STEP)
                target_action_values_np[not_done_idx] += last_values_np

        target_action_values_v = float32_preprocessor(target_action_values_np).to(self.device)

        # states_v.shape: [128, 3]
        # actions_v.shape: [128, 1]
        # target_action_values_v.shape: [128]
        return states_v, actions_v, target_action_values_v

    def get_advantage_and_target_action_values(self, trajectory, values_v, device):
        """
        By trajectory calculate advantage and 1-step target action value
        :param trajectory: trajectory list
        :return: tuple with advantage numpy array and reference values
        """
        values = values_v.squeeze().data.cpu().numpy()

        # generalized advantage estimator: smoothed version of the advantage
        last_gae = 0.0
        result_advantages = []
        result_target_action_values = []
        for value, next_value, exp in zip(reversed(values[:-1]), reversed(values[1:]), reversed(trajectory[:-1])):
            if exp.done:
                delta = exp.reward - value
                last_gae = delta
            else:
                delta = exp.reward + self.params.GAMMA * next_value - value
                last_gae = delta + self.params.GAMMA * self.params.PPO_GAE_LAMBDA * last_gae

            result_advantages.append(last_gae)
            result_target_action_values.append(last_gae + value)

        advantage_v = float32_preprocessor(list(reversed(result_advantages)))
        target_action_value_v = float32_preprocessor(list(reversed(result_target_action_values)))
        return advantage_v.to(device), target_action_value_v.to(device)
