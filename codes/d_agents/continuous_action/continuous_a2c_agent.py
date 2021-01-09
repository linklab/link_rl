import math

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn.utils as nn_utils
from torch.distributions import Normal

from codes.d_agents.a0_base_agent import BaseAgent, float32_preprocessor
from codes.e_utils import rl_utils, replay_buffer


class AgentContinuousA2C(BaseAgent):
    """
    """
    def __init__(
            self, worker_id, input_shape, num_outputs, action_selector, action_min, action_max, params,
            preprocessor=float32_preprocessor, device="cpu"
    ):
        super(AgentContinuousA2C, self).__init__()
        self.__name__ = "AgentContinuousA2C"
        self.device = device
        self.preprocessor = preprocessor
        self.action_selector = action_selector
        self.worker_id = worker_id
        self.params = params
        self.device = device
        self.action_min = action_min
        self.action_max = action_max

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

        self.buffer = replay_buffer.ExperienceReplayBuffer(experience_source=None, buffer_size=self.params.BATCH_SIZE)

    def __call__(self, states, critics=None):
        if self.preprocessor is not None:
            states = self.preprocessor(states)
            if torch.is_tensor(states):
                states = states.to(self.device)

        if len(states) == 1:
            self.model.eval()
        else:
            self.model.train()

        mu_v, var_v, values_v = self.model(states)
        actions = self.action_selector(mu_v, var_v, self.action_min, self.action_max)
        critics = [values_v.data.squeeze().cpu().numpy()]
        return actions, critics

    def train_net(self, step_idx):
        batch = self.buffer.sample(self.params.BATCH_SIZE)

        # states_v.shape: (32, 3)
        # actions_v.shape: (32,)
        # target_action_values_v.shape: (32,)
        states_v, actions_v, target_action_values_v = self.unpack_batch_for_actor_critic(
            batch, self.model, self.params, device=self.device
        )

        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()

        # mu_v.shape: (32, 1)
        # var_v.shape: (32, 1)
        # value_v.shape; (32, 1)
        mu_v, var_v, value_v = self.model(states_v)

        # advantage_v.shape: (32, 1)
        advantage_v = target_action_values_v.unsqueeze(dim=-1) - value_v.detach()
        log_prob_actions_v = advantage_v * self.calc_log_prob(mu_v, var_v, actions_v)
        loss_actor_v = -1.0 * log_prob_actions_v.mean()

        entropy_v = -1.0 * (torch.log(2.0 * math.pi * var_v) + 1) / 2
        loss_entropy_v = self.params.ENTROPY_BETA * entropy_v.mean()

        # loss_actor_v를 작아지도록 만듦 --> batch_log_prob_actions_v.mean()가 커지도록 만듦
        # loss_entropy_v를 작아지도록 만듦 --> entropy_v가 커지도록 만듦
        # print(loss_critic_v, loss_actor_v, loss_entropy_v)
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

    # @staticmethod
    # def calc_log_prob(mu_v, var_v, actions_v):
    #     p1 = - ((mu_v - actions_v) ** 2) / (2 * var_v.clamp(min=1e-3))
    #     p2 = - torch.log(torch.sqrt(2 * math.pi * var_v))
    #     return p1 + p2

    @staticmethod
    def calc_log_prob(mu_v, var_v, actions_v):
        n = Normal(mu_v, torch.sqrt(var_v))
        log_prob = n.log_prob(actions_v)
        return log_prob
