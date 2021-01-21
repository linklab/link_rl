import math

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn.utils as nn_utils
from torch.distributions import Normal, MultivariateNormal

from codes.d_agents.a0_base_agent import BaseAgent, float32_preprocessor
from codes.d_agents.on_policy.on_policy_agent import OnPolicyAgent
from codes.e_utils import rl_utils, replay_buffer
from codes.e_utils.names import DeepLearningModelName


class AgentContinuousA2C(OnPolicyAgent):
    """
    """
    def __init__(
            self, worker_id, input_shape, num_outputs, action_selector, action_min, action_max, params,
            preprocessor=float32_preprocessor, device="cpu"
    ):
        super(AgentContinuousA2C, self).__init__(params, device)
        self.__name__ = "AgentContinuousA2C"
        self.device = device
        self.preprocessor = preprocessor
        self.action_selector = action_selector
        self.worker_id = worker_id
        self.action_min = action_min
        self.action_max = action_max

        assert params.DEEP_LEARNING_MODEL == DeepLearningModelName.STOCHASTIC_CONTINUOUS_ACTOR_CRITIC_MLP
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

        mu_v, var_v = self.model.base.actor(states)
        actions = self.action_selector(mu_v, var_v, self.action_min, self.action_max)
        critics = torch.zeros(size=mu_v.size())

        return actions, critics

    def train_net(self, step_idx):
        batch = self.buffer.sample(batch_size=None)

        # states_v.shape: (32, 3)
        # actions_v.shape: (32, 1)
        # target_action_values_v.shape: (32,)
        states_v, actions_v, target_action_values_v = self.unpack_batch_for_actor_critic(batch, self.model, self.params)

        batch.clear()

        # self.optimizer.zero_grad()
        self.critic_optimizer.zero_grad()

        # mu_v.shape: (32, 1)
        # var_v.shape: (32, 1)
        # value_v.shape; (32, 1)
        mu_v, var_v, value_v = self.model(states_v)

        # Critic Optimization
        loss_critic_v = F.mse_loss(input=value_v.squeeze(-1), target=target_action_values_v.detach())
        loss_critic_v.backward(retain_graph=True)
        #nn_utils.clip_grad_norm_(self.model.base.critic.parameters(), self.params.CLIP_GRAD)
        self.critic_optimizer.step()

        # Actor Optimization
        self.actor_optimizer.zero_grad()

        # advantage_v.shape: (32, 1)
        advantage_v = target_action_values_v.unsqueeze(dim=-1).detach() - value_v.detach()

        # covariance_matrix = torch.diag_embed(var_v).to(self.device)
        # dist = MultivariateNormal(loc=mu_v, covariance_matrix=covariance_matrix)
        # log_pi_action_v = advantage_v * dist.log_prob(actions_v).unsqueeze(-1)
        dist = Normal(loc=mu_v, scale=torch.sqrt(var_v))
        log_pi_action_v = advantage_v * dist.log_prob(actions_v)

        print(advantage_v.size(), dist.log_prob(actions_v).size(), log_pi_action_v.size())

        # log_pi_v = advantage_v * self.calc_log_pi(mu_v, self.model.base.actor.logstd, actions_v)
        loss_actor_v = -1.0 * log_pi_action_v.mean()

        loss_entropy_v = -1.0 * self.params.PPO_ENTROPY_WEIGHT * dist.entropy().mean()

        #print(loss_actor_v, loss_entropy_v)

        # entropy_v = -1.0 * (torch.log(2.0 * math.pi * torch.exp(self.model.base.actor.logstd)) + 1) / 2
        # loss_entropy_v = self.params.ENTROPY_BETA * entropy_v.mean()

        # loss_actor_v를 작아지도록 만듦 --> batch_log_pi_v.mean()가 커지도록 만듦
        # loss_entropy_v를 작아지도록 만듦 --> entropy_v가 커지도록 만듦
        # print(loss_critic_v, loss_actor_v, loss_entropy_v)
        loss_actor_and_entropy_v = loss_actor_v + loss_entropy_v

        loss_actor_and_entropy_v.backward()
        #nn_utils.clip_grad_norm_(self.model.base.actor.parameters(), self.params.CLIP_GRAD)
        self.actor_optimizer.step()

        gradients = self.model.get_gradients_for_current_parameters()

        return gradients, loss_critic_v.item(), loss_actor_v.item() * -1.0

    # @staticmethod
    # def calc_log_pi(mu_v, logstd_v, actions_v):
    #     p1 = - ((mu_v - actions_v) ** 2) / (2 * torch.exp(logstd_v).clamp(min=1e-3))
    #     p2 = - torch.log(torch.sqrt(2 * math.pi * torch.exp(logstd_v)))
    #     return p1 + p2

    # @staticmethod
    # def calc_log_pi(mu_v, var_v, actions_v):
    #     # https://pytorch.org/docs/stable/generated/torch.clamp.html, clamp: 단단히 고정시키다.
    #     p1 = - ((mu_v - actions_v) ** 2) / (2 * var_v.clamp(min=1e-3))
    #     p2 = - torch.log(torch.sqrt(2 * math.pi * var_v))
    #     return p1 + p2

    # @staticmethod
    # def calc_log_pi(mu_v, var_v, actions_v):
    #     n = Normal(mu_v, torch.sqrt(var_v))
    #     log_pi = n.log_pi(actions_v)
    #     return log_pi
