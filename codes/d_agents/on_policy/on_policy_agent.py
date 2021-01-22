import numpy as np
import torch

from codes.d_agents.a0_base_agent import BaseAgent, float32_preprocessor
from codes.e_utils import replay_buffer


class OnPolicyAgent(BaseAgent):
    """
    Abstract Agent interface
    """
    def __init__(self, params, device):
        super(OnPolicyAgent, self).__init__()
        self.params = params
        self.device = device

        self.buffer = replay_buffer.ExperienceReplayBuffer(
            experience_source=None, buffer_size=self.params.REPLAY_BUFFER_SIZE
        )

    def unpack_batch_for_actor_critic(self, batch, net, params):
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
        actions_v = float32_preprocessor(actions).to(self.device)

        # handle rewards
        target_action_values_np = np.array(rewards, dtype=np.float32)

        if not_done_idx:
            last_states_v = torch.FloatTensor(np.array(last_states, copy=False)).to(self.device)
            last_values_v = net.base.forward_critic(last_states_v)
            last_values_np = last_values_v.data.cpu().numpy()[:, 0] * params.GAMMA ** params.N_STEP
            target_action_values_np[not_done_idx] += last_values_np

        target_action_values_v = float32_preprocessor(target_action_values_np).to(self.device)

        return states_v, actions_v, target_action_values_v