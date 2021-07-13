from abc import abstractmethod

import numpy as np
import torch

from codes.c_models.base_model import RNNModel
from codes.d_agents.a0_base_agent import BaseAgent, float32_preprocessor
from codes.e_utils import replay_buffer
from codes.e_utils.names import RLAlgorithmName


class OffPolicyAgent(BaseAgent):
    """
    Abstract Agent interface
    """
    def __init__(self, worker_id, action_shape, params, device):
        super(OffPolicyAgent, self).__init__(worker_id, action_shape, params, device)

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
        state, actions, rewards, dones, last_states, agent_states = [], [], [], [], [], []

        if isinstance(self.model, RNNModel):
            actor_hidden_states = []
            if self.params.RL_ALGORITHM in [RLAlgorithmName.DDPG_V0]:
                critic_hidden_states = []
                critic_1_hidden_states = None
                critic_2_hidden_states = None
            elif self.params.RL_ALGORITHM in [RLAlgorithmName.TD3_V0]:
                critic_hidden_states = None
                critic_1_hidden_states = []
                critic_2_hidden_states = []
            else:
                raise ValueError()
        else:
            actor_hidden_states = critic_hidden_states = critic_1_hidden_states = critic_2_hidden_states = None

        for exp in batch:
            state.append(np.array(exp.state, copy=False))
            actions.append(exp.action)
            rewards.append(exp.reward)
            dones.append(exp.last_state is None)
            if exp.last_state is None:
                last_states.append(exp.state)  # the result will be masked anyway
            else:
                last_states.append(np.array(exp.last_state, copy=False))

            if isinstance(self.model, RNNModel):
                actor_hidden_states.append(exp.agent_state.actor_hidden_state)
                if self.params.RL_ALGORITHM in [RLAlgorithmName.DDPG_V0]:
                    critic_hidden_states.append(exp.agent_state.critic_hidden_state)
                elif self.params.RL_ALGORITHM in [RLAlgorithmName.TD3_V0]:
                    critic_1_hidden_states.append(exp.agent_state.critic_1_hidden_state)
                    critic_2_hidden_states.append(exp.agent_state.critic_2_hidden_state)

        states_v = float32_preprocessor(state).to(self.device)
        actions_v = float32_preprocessor(actions).to(self.device)
        rewards_v = float32_preprocessor(rewards).to(self.device)
        last_states_v = float32_preprocessor(last_states).to(self.device)
        dones_t = torch.BoolTensor(dones).to(self.device)

        if isinstance(self.model, RNNModel):
            actor_hidden_states_v = float32_preprocessor(actor_hidden_states).to(self.device)
            if self.params.RL_ALGORITHM in [RLAlgorithmName.DDPG_V0]:
                critic_hidden_states_v = float32_preprocessor(critic_hidden_states).to(self.device)
                return states_v, actions_v, rewards_v, dones_t, last_states_v, agent_states, actor_hidden_states_v, \
                       critic_hidden_states_v

            elif self.params.RL_ALGORITHM in [RLAlgorithmName.TD3_V0]:
                critic_1_hidden_states_v = float32_preprocessor(critic_1_hidden_states).to(self.device)
                critic_2_hidden_states_v = float32_preprocessor(critic_2_hidden_states).to(self.device)
                return states_v, actions_v, rewards_v, dones_t, last_states_v, agent_states, actor_hidden_states_v, \
                       critic_1_hidden_states_v, critic_2_hidden_states_v

            else:
                pass
        else:
            return states_v, actions_v, rewards_v, dones_t, last_states_v, agent_states



