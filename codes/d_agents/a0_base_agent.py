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

    def unpack_batch_for_actor_critic(
            self, batch, target_model=None, sac_base_model=None, alpha=None, params=None
    ):
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
            elif self.params.RL_ALGORITHM in [RLAlgorithmName.CONTINUOUS_SAC_V0]:
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
                elif self.params.RL_ALGORITHM in [RLAlgorithmName.CONTINUOUS_SAC_V0]:
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
            elif self.params.RL_ALGORITHM in [RLAlgorithmName.CONTINUOUS_SAC_V0]:
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
                last_values_v, _ = target_model.forward_critic(last_states_v, critic_hidden_states_v)
                last_values_np = last_values_v.detach().numpy()[:, 0] * (params.GAMMA ** last_steps_v)
                target_action_values_np[not_done_idx] += last_values_np

            elif self.params.RL_ALGORITHM in [RLAlgorithmName.DISCRETE_SAC_V0, RLAlgorithmName.CONTINUOUS_SAC_V0]:
                if self.params.RL_ALGORITHM in [RLAlgorithmName.CONTINUOUS_SAC_V0]:
                    last_mu_v, last_logstd_v, _ = sac_base_model.forward_actor(last_states_v)
                    dist = Normal(loc=last_mu_v, scale=torch.exp(last_logstd_v))

                    last_actions_v = dist.sample()
                    last_logprob_v = dist.log_prob(last_actions_v).sum(dim=-1, keepdim=True)

                    last_q_1_v, last_q_2_v = target_model.base.twinq(last_states_v, last_actions_v)
                    last_q_np = torch.min(last_q_1_v, last_q_2_v).detach().numpy()[:, 0] * (params.GAMMA ** last_steps_v)
                    last_logprob_v = alpha * last_logprob_v

                    # last_q_np.shape: (128,)
                    # entropy_v.squeeze(-1).detach().numpy().shape: (128,)
                    last_q_np -= last_logprob_v.squeeze(-1).detach().numpy()

                else:
                    # probs.shape: torch.Size([32, 2])
                    probs, _ = sac_base_model.forward_actor(last_states_v)
                    z = (probs == 0.0).float() * 1e-8

                    last_logprob_v = torch.log(probs + z)

                    last_q_1_v, last_q_2_v = target_model.base.twinq(last_states_v)
                    last_q_np = torch.min(last_q_1_v, last_q_2_v).detach().numpy() * np.expand_dims(params.GAMMA ** last_steps_v, axis=-1)
                    last_logprob_v = alpha * last_logprob_v
                    last_q_np = probs * (last_q_np - last_logprob_v)
                    last_q_np = last_q_np.squeeze(-1).detach().numpy()

                target_action_values_np[not_done_idx] += last_q_np
            else:
                raise ValueError()

        target_action_values_v = float32_preprocessor(target_action_values_np).to(self.device)

        # states_v.shape: [128, 3]
        # actions_v.shape: [128, 1]
        # target_action_values_v.shape: [128]

        if isinstance(self.model, RNNModel):
            if self.params.RL_ALGORITHM in [RLAlgorithmName.CONTINUOUS_SAC_V0]:
                return states_v, actions_v, target_action_values_v, actor_hidden_states_v, \
                       critic_1_hidden_states_v, critic_2_hidden_states_v
            else:
                return states_v, actions_v, target_action_values_v, actor_hidden_states_v, critic_hidden_states_v
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
