from abc import abstractmethod

import numpy as np
import torch

from codes.d_agents.a0_base_agent import BaseAgent, float32_preprocessor
from codes.e_utils import replay_buffer


class OffPolicyAgent(BaseAgent):
    """
    Abstract Agent interface
    """
    def __init__(self, worker_id, params, action_shape, action_min, action_max, device):
        super(OffPolicyAgent, self).__init__(worker_id, params, action_shape, action_min, action_max, device)

        if hasattr(self.params, "PER_PROPORTIONAL") and self.params.PER_PROPORTIONAL:
            self.buffer = replay_buffer.PrioritizedReplayBuffer(
                experience_source=None, buffer_size=self.params.REPLAY_BUFFER_SIZE,
                n_step=self.params.N_STEP, beta_start=0.4, beta_frames=self.params.MAX_GLOBAL_STEP
            )
        elif hasattr(self.params, "PER_RANK_BASED") and self.params.PER_RANK_BASED:
            self.buffer = replay_buffer.RankBasedPrioritizedReplayBuffer(
                experience_source=None, buffer_size=self.params.REPLAY_BUFFER_SIZE,
                params=self.params, alpha=0.7, beta_start=0.5, beta_frames=self.params.MAX_GLOBAL_STEP
            )
        else:
            self.buffer = replay_buffer.ExperienceReplayBuffer(
                experience_source=None, buffer_size=self.params.REPLAY_BUFFER_SIZE
            )

    def train_off_policy(self, step_idx):
        train_results = self.on_train(step_idx)
        return train_results

    @abstractmethod
    def on_train(self, step_idx):
        raise NotImplementedError

    def unpack_batch(self, batch):
        states, actions, rewards, dones, last_states = [], [], [], [], []

        for exp in batch:
            states.append(np.array(exp.state, copy=False))
            actions.append(exp.action)
            rewards.append(exp.reward)
            dones.append(exp.last_state is None)
            if exp.last_state is None:
                last_states.append(exp.state)  # the result will be masked anyway
            else:
                last_states.append(np.array(exp.last_state, copy=False))

        states_v = float32_preprocessor(states).to(self.device)
        actions_v = float32_preprocessor(actions).to(self.device)
        rewards_v = float32_preprocessor(rewards).to(self.device)
        last_states_v = float32_preprocessor(last_states).to(self.device)
        dones_t = torch.BoolTensor(dones).to(self.device)

        return states_v, actions_v, rewards_v, dones_t, last_states_v

