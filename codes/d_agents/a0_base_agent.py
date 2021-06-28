from abc import abstractmethod
from collections import namedtuple

import numpy as np
import torch

from codes.c_models.base_model import RNNModel
from codes.c_models.continuous_action.continuous_action_model import ContinuousActionModel
from codes.c_models.discrete_action.discrete_action_model import DiscreteActionModel
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

    def continuous_sac_call(self, state):
        state = self.preprocess(state)

        if len(state) == 1:
            self.model.eval()
        else:
            self.model.train()

        if self.agent_mode == AgentMode.TRAIN:
            with torch.no_grad():
                mu_v, logstd_v = self.model.base.actor(state)
                actions = self.train_action_selector(mu_v=mu_v, logstd_v=logstd_v)
        else:
            with torch.no_grad():
                mu_v, _ = self.test_model.base.actor(state)
                actions = self.test_and_play_action_selector(mu_v=mu_v, logstd_v=None)

        critics = torch.zeros(size=mu_v.size())

        return actions, critics

    def unpack_batch_for_actor_critic(self, batch, model, params, alpha=None):
        """
        Convert batch into training tensors
        :param batch:
        :param model:
        :return: state variable, actions tensor, target values variable
        """
        states, actions, rewards, not_done_idx, last_states, last_steps = [], [], [], [], [], []

        if isinstance(self.model, RNNModel):
            actor_hidden_states = []
            if self.params.RL_ALGORITHM in [RLAlgorithmName.DISCRETE_A2C_V0, RLAlgorithmName.CONTINUOUS_A2C_V0]:
                critic_hidden_states = []
                critic_1_hidden_states = None
                critic_2_hidden_states = None
            elif self.params.RL_ALGORITHM in [RLAlgorithmName.SAC_V0]:
                critic_hidden_states = None
                critic_1_hidden_states = []
                critic_2_hidden_states = []
            else:
                raise ValueError()
        else:
            actor_hidden_states = critic_hidden_states = critic_1_hidden_states = critic_2_hidden_states = None

        for idx, exp in enumerate(batch):
            states.append(np.array(exp.state, copy=False))
            actions.append(exp.action)
            rewards.append(exp.reward)

            if exp.last_state is not None:
                not_done_idx.append(idx)
                last_states.append(np.array(exp.last_state, copy=False))
                last_steps.append(exp.last_step)

            if isinstance(self.model, RNNModel):
                actor_hidden_states.append(exp.agent_state.actor_hidden_state)
                if self.params.RL_ALGORITHM in [RLAlgorithmName.DISCRETE_A2C_V0, RLAlgorithmName.CONTINUOUS_A2C_V0]:
                    critic_hidden_states.append(exp.agent_state.critic_hidden_state)
                elif self.params.RL_ALGORITHM in [RLAlgorithmName.SAC_V0]:
                    critic_1_hidden_states.append(exp.agent_state.critic_1_hidden_state)
                    critic_2_hidden_states.append(exp.agent_state.critic_2_hidden_state)

        states_v = float32_preprocessor(states).to(self.device)
        actions_v = self.convert_action_to_torch_tensor(actions, self.device)
        last_steps_v = np.asarray(last_steps)

        if isinstance(self.model, RNNModel):
            actor_hidden_states_v = float32_preprocessor(actor_hidden_states).to(self.device)
            if self.params.RL_ALGORITHM in [RLAlgorithmName.DISCRETE_A2C_V0, RLAlgorithmName.CONTINUOUS_A2C_V0]:
                critic_hidden_states_v = float32_preprocessor(critic_hidden_states).to(self.device)
                critic_1_hidden_states_v = None
                critic_2_hidden_states_v = None
            elif self.params.RL_ALGORITHM in [RLAlgorithmName.SAC_V0]:
                critic_hidden_states_v = None
                critic_1_hidden_states_v = float32_preprocessor(critic_1_hidden_states).to(self.device)
                critic_2_hidden_states_v = float32_preprocessor(critic_2_hidden_states).to(self.device)
            else:
                raise ValueError()
        else:
            actor_hidden_states_v = critic_hidden_states_v = critic_1_hidden_states_v = critic_2_hidden_states_v = None

        # handle rewards
        target_action_values_np = np.array(rewards, dtype=np.float32)

        if not_done_idx:
            last_states_v = torch.FloatTensor(np.array(last_states, copy=False)).to(self.device)
            if self.params.RL_ALGORITHM in [RLAlgorithmName.DISCRETE_A2C_V0, RLAlgorithmName.CONTINUOUS_A2C_V0]:
                last_values_v, _ = model.forward_critic(last_states_v, critic_hidden_states_v)
                last_values_np = last_values_v.data.cpu().numpy()[:, 0] * (params.GAMMA ** last_steps_v)
                target_action_values_np[not_done_idx] += last_values_np
            elif self.params.RL_ALGORITHM in [RLAlgorithmName.SAC_V0]:
                last_actions_v, last_entropies_v = model.sample(last_states_v)
                last_q_1_v, last_q_2_v = model.base.twinq(last_states_v, last_actions_v)
                last_q_v = torch.min(last_q_1_v, last_q_2_v) * (params.GAMMA ** last_steps_v)
                last_q_v += alpha * last_entropies_v
                last_q_v = last_q_v.squeeze(-1)
                target_action_values_np[not_done_idx] += last_q_v.data.cpu().numpy()
            else:
                raise ValueError()

        target_action_values_v = float32_preprocessor(target_action_values_np).to(self.device)

        # states_v.shape: [128, 3]
        # actions_v.shape: [128, 1]
        # target_action_values_v.shape: [128]

        if isinstance(self.model, RNNModel):
            if self.params.RL_ALGORITHM in [RLAlgorithmName.DISCRETE_A2C_V0, RLAlgorithmName.CONTINUOUS_A2C_V0]:
                return states_v, actions_v, target_action_values_v, actor_hidden_states_v, critic_hidden_states_v
            elif self.params.RL_ALGORITHM in [RLAlgorithmName.SAC_V0]:
                return states_v, actions_v, target_action_values_v, actor_hidden_states_v, \
                       critic_1_hidden_states_v, critic_2_hidden_states_v
            else:
                raise ValueError()
        else:
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

    def convert_action_to_torch_tensor(self, values, device):
        if isinstance(self.model, DiscreteActionModel):
            actions_v = long64_preprocessor(values).to(device)
        elif isinstance(self.model, ContinuousActionModel):
            actions_v = float32_preprocessor(values).to(device)
        else:
            raise ValueError()

        return actions_v
