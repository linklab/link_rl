import numpy as np
import torch
import torch.nn.functional as F
import torch.nn.utils as nn_utils

from codes.d_agents.a0_base_agent import BaseAgent, float32_preprocessor
from codes.e_utils import rl_utils, replay_buffer


class AgentDiscreteA2C(BaseAgent):
    """
    """
    def __init__(
            self, worker_id, input_shape, num_outputs, action_selector, params,
            preprocessor=float32_preprocessor, device="cpu"
    ):
        super(AgentDiscreteA2C, self).__init__()
        self.__name__ = "AgentDiscreteA2C"
        self.device = device
        self.preprocessor = preprocessor
        self.action_selector = action_selector
        self.worker_id = worker_id
        self.params = params
        self.device = device

        self.model = rl_utils.get_rl_model(
            worker_id=worker_id, input_shape=input_shape, num_outputs=num_outputs, params=params, device=self.device
        )

        self.actor_optimizer = rl_utils.get_optimizer(
            parameters=self.model.base.actor.parameters(),
            learning_rate=self.params.ACTOR_LEARNING_RATE,
            params=params
        )

        self.critic_optimizer = rl_utils.get_optimizer(
            parameters=self.model.base.critic.parameters(),
            learning_rate=self.params.LEARNING_RATE,
            params=params
        )

        self.buffer = replay_buffer.ExperienceReplayBuffer(
            experience_source=None, buffer_size=self.params.BATCH_SIZE
        )

    def __call__(self, states, critics=None):
        if self.preprocessor:
            states = self.preprocessor(states)
            if torch.is_tensor(states):
                states = states.to(self.device)

        probs_v, values_v = self.model(states)

        probs_v = F.softmax(probs_v, dim=1)

        probs = probs_v.data.cpu().numpy()
        actions = np.array(self.action_selector(probs))
        critics = [values_v.data.squeeze().cpu().numpy()]
        return actions, critics

    def train_net(self, step_idx):
        batch = self.buffer.sample(self.params.BATCH_SIZE)

        batch_states_v, batch_actions_v, batch_target_action_values_v = self.unpack_batch_for_a2c(
            batch, self.model, self.params, device=self.device
        )

        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()

        batch_logits_v, batch_value_v = self.model(batch_states_v)

        batch_advantage_v = batch_target_action_values_v - batch_value_v.squeeze(-1).detach()
        batch_log_prob_v = F.log_softmax(batch_logits_v, dim=1)
        batch_log_prob_actions_v = batch_log_prob_v[range(self.params.BATCH_SIZE), batch_actions_v]
        batch_log_prob_actions_v = batch_advantage_v * batch_log_prob_actions_v
        loss_actor_v = -1.0 * batch_log_prob_actions_v.mean()

        batch_prob_v = F.softmax(batch_logits_v, dim=1)
        entropy_v = -(batch_prob_v * batch_log_prob_v).sum(dim=1).mean()
        loss_entropy_v = -1.0 * self.params.ENTROPY_BETA * entropy_v

        # loss_actor_v를 작아지도록 만듦 --> batch_log_prob_actions_v.mean()가 커지도록 만듦
        # loss_entropy_v를 작아지도록 만듦 --> entropy_v가 커지도록 만듦
        loss_actor_and_entropy_v = loss_actor_v + loss_entropy_v

        loss_actor_and_entropy_v.backward(retain_graph=True)
        nn_utils.clip_grad_norm_(self.model.base.actor.parameters(), self.params.CLIP_GRAD)
        self.actor_optimizer.step()

        loss_critic_v = F.mse_loss(batch_value_v.squeeze(-1), batch_target_action_values_v)
        loss_critic_v.backward()
        nn_utils.clip_grad_norm_(self.model.base.critic.parameters(), self.params.CLIP_GRAD)
        self.critic_optimizer.step()

        gradients = self.model.get_gradients_for_current_parameters()

        batch.clear()

        return gradients, loss_critic_v.item(), loss_actor_v.item() * -1.0

    def unpack_batch_for_a2c(self, batch, net, params, device='cpu'):
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
