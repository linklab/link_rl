from abc import abstractmethod

import numpy as np
import torch

from codes.d_agents.a0_base_agent import BaseAgent, float32_preprocessor, long64_preprocessor
from codes.e_utils import replay_buffer
from codes.e_utils.names import RLAlgorithmName


class OnPolicyAgent(BaseAgent):
    """
    Abstract Agent interface
    """
    def __init__(self, train_action_selector, test_and_play_action_selector, params, device):
        super(OnPolicyAgent, self).__init__(train_action_selector, test_and_play_action_selector, params, device)

        self.buffer = replay_buffer.ExperienceReplayBuffer(
            experience_source=None, buffer_size=self.params.REPLAY_BUFFER_SIZE
        )

    def unpack_batch_for_actor_critic(self, batch, net, params, discrete=False):
        """
        Convert batch into training tensors
        :param batch:
        :param net:
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
        elif params.RL_ALGORITHM in [RLAlgorithmName.CONTINUOUS_A2C_V0, RLAlgorithmName.CONTINUOUS_PPO_V0]:
            actions_v = float32_preprocessor(actions).to(self.device)
        else:
            raise ValueError()

        # handle rewards
        target_action_values_np = np.array(rewards, dtype=np.float32)

        if not_done_idx:
            last_states_v = torch.FloatTensor(np.array(last_states, copy=False)).to(self.device)
            last_values_v = net.base.forward_critic(last_states_v)
            last_values_np = last_values_v.data.cpu().numpy()[:, 0] * (params.GAMMA ** params.N_STEP)
            target_action_values_np[not_done_idx] += last_values_np

        target_action_values_v = float32_preprocessor(target_action_values_np).to(self.device)

        return states_v, actions_v, target_action_values_v

    def get_advantage_and_target_action_values(self, trajectory, values_v, device):
        """
        By trajectory calculate advantage and 1-step target action value
        :param trajectory: trajectory list
        :param critic_model: critic deep learning network
        :param states_v: states tensor
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