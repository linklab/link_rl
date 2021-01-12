import copy

import numpy as np
import torch


def float32_preprocessor(states):
    np_states = np.array(states, dtype=np.float32)
    return torch.tensor(np_states)


class BaseAgent:
    """
    Abstract Agent interface
    """
    def __init__(self):
        self.buffer = None
        pass

    def initial_agent_state(self):
        """
        Should create initial empty state for the agent. It will be called for the start of the episode
        :return: Anything agent want to remember
        """
        return None

    def initial_agent_state(self):
        return 0.0

    def set_experience_source_to_buffer(self, experience_source):
        self.buffer.set_experience_source(experience_source)

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

    def unpack_batch_for_actor_critic(self, batch, net, params, device='cpu'):
        """
        Convert batch into training tensors
        :param batch:
        :param net:
        :return: states variable, actions tensor, target values variable
        """
        states, actions, rewards, not_done_idx, last_states = [], [], [], [], []

        for idx, exp in enumerate(batch):
            states.append(np.array(exp.state, copy=False))
            actions.append(int(exp.action))
            rewards.append(exp.reward)
            if exp.last_state is not None:
                not_done_idx.append(idx)
                last_states.append(np.array(exp.last_state, copy=False))

        states_v = torch.FloatTensor(np.array(states, copy=False)).to(device)
        actions_v = torch.LongTensor(actions).to(device)

        # handle rewards
        target_action_values_np = np.array(rewards, dtype=np.float32)

        if not_done_idx:
            last_states_v = torch.FloatTensor(np.array(last_states, copy=False)).to(device)
            last_values_v = net.base.forward_critic(last_states_v)
            last_values_np = last_values_v.data.cpu().numpy()[:, 0] * params.GAMMA ** params.N_STEP
            target_action_values_np[not_done_idx] += last_values_np

        target_action_values_v = torch.FloatTensor(target_action_values_np).to(device)

        return states_v, actions_v, target_action_values_v


class TargetNet:
    """
    Wrapper around model which provides copy of it instead of trained weights
    """
    def __init__(self, model):
        self.model = model
        self.target_model = copy.deepcopy(model)

    def sync(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def alpha_sync(self, alpha):
        """
        Blend params of target net with params from the model
        :param alpha:
        """
        assert isinstance(alpha, float)
        assert 0.0 < alpha <= 1.0
        state = self.model.state_dict()
        tgt_state = self.target_model.state_dict()
        for k, v in state.items():
            tgt_state[k] = tgt_state[k] * alpha + (1.0 - alpha) * v
        self.target_model.load_state_dict(tgt_state)
