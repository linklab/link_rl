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
        critics = values_v.data.cpu().numpy()
        return actions, critics

    def train_net(self, step_idx):
        batch = self.buffer.sample(self.params.BATCH_SIZE)

        states_v, actions_v, target_action_values_v = self.unpack_batch_for_actor_critic(
            batch, self.model, self.params, device=self.device
        )

        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()

        logits_v, value_v = self.model(states_v)

        advantage_v = target_action_values_v - value_v.squeeze(-1).detach()
        log_prob_v = F.log_softmax(logits_v, dim=1)
        log_prob_actions_v = log_prob_v[range(self.params.BATCH_SIZE), actions_v]
        log_prob_actions_v = advantage_v * log_prob_actions_v
        loss_actor_v = -1.0 * log_prob_actions_v.mean()

        prob_v = F.softmax(logits_v, dim=1)
        entropy_v = -(prob_v * log_prob_v).sum(dim=1).mean()
        loss_entropy_v = -1.0 * self.params.ENTROPY_BETA * entropy_v

        # loss_actor_v를 작아지도록 만듦 --> batch_log_prob_actions_v.mean()가 커지도록 만듦
        # loss_entropy_v를 작아지도록 만듦 --> entropy_v가 커지도록 만듦
        loss_actor_and_entropy_v = loss_actor_v + loss_entropy_v

        loss_actor_and_entropy_v.backward(retain_graph=True)
        nn_utils.clip_grad_norm_(self.model.base.actor.parameters(), self.params.CLIP_GRAD)
        self.actor_optimizer.step()

        loss_critic_v = F.mse_loss(value_v.squeeze(-1), target_action_values_v)
        loss_critic_v.backward()
        nn_utils.clip_grad_norm_(self.model.base.critic.parameters(), self.params.CLIP_GRAD)
        self.critic_optimizer.step()

        gradients = self.model.get_gradients_for_current_parameters()

        batch.clear()

        return gradients, loss_critic_v.item(), loss_actor_v.item() * -1.0

