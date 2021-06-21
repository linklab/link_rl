import math

import torch
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np

from codes.c_models.continuous_action.stochastic_continuous_actor_critic_model import StochasticContinuousActorCriticModel
from codes.d_agents.on_policy.a2c.a2c_agent import AgentA2C
from codes.d_agents.on_policy.on_policy_action_selector import ContinuousNormalActionSelector
from codes.e_utils import rl_utils
from codes.e_utils.names import DeepLearningModelName


class AgentContinuousA2C(AgentA2C):
    """
    """
    def __init__(
            self, worker_id, input_shape, action_shape, num_outputs, params, device
    ):
        assert params.DEEP_LEARNING_MODEL in [
            DeepLearningModelName.STOCHASTIC_CONTINUOUS_ACTOR_CRITIC_MLP
        ]
        super(AgentContinuousA2C, self).__init__(worker_id, input_shape, action_shape, num_outputs, params, device)

        self.__name__ = "AgentContinuousA2C"
        self.train_action_selector = ContinuousNormalActionSelector()
        self.test_and_play_action_selector = ContinuousNormalActionSelector()

        self.model = StochasticContinuousActorCriticModel(
            worker_id=worker_id,
            input_shape=input_shape,
            num_outputs=num_outputs,
            params=params,
            device=device
        ).to(device)

        self.test_model = StochasticContinuousActorCriticModel(
            worker_id=worker_id,
            input_shape=input_shape,
            num_outputs=num_outputs,
            params=params,
            device=device
        ).to(device)

        self.actor_optimizer = rl_utils.get_optimizer(
            parameters=self.model.base.actor_params,
            learning_rate=self.params.ACTOR_LEARNING_RATE,
            params=params
        )

        self.critic_optimizer = rl_utils.get_optimizer(
            parameters=self.model.base.critic_params,
            learning_rate=self.params.LEARNING_RATE,
            params=params
        )

    def __call__(self, states, critics=None):
        return self.continuous_stochastic_call(states, critics)

    def train(self, step_idx):
        batch = self.buffer.sample(batch_size=None)

        # states_v.shape: (32, 3)
        # actions_v.shape: (32, 1)
        # target_action_values_v.shape: (32,)
        states_v, actions_v, target_action_values_v = self.unpack_batch_for_actor_critic(batch, self.model, self.params)

        # mu_v.shape: (32, 1)
        # var_v.shape: (32, 1)
        # value_v.shape; (32, 1)
        mu_v, logstd_v, value_v = self.model(states_v)
        # print(mu_v.shape, logstd_v.shape, value_v.shape, "##############")

        # Critic Optimization
        loss_critic_v = F.mse_loss(input=value_v.squeeze(-1), target=target_action_values_v.detach())

        self.critic_optimizer.zero_grad()
        loss_critic_v.backward(retain_graph=True)
        self.critic_optimizer.step()

        # Actor Optimization
        # advantage_v.shape: (32,)
        advantage_v = target_action_values_v - value_v.squeeze(-1)

        dist = Normal(loc=mu_v, scale=torch.exp(logstd_v))
        reinforced_log_pi_action_v = dist.log_prob(value=actions_v) * advantage_v.unsqueeze(dim=-1).detach()

        # reinforced_log_pi_action_v = self.calc_logprob(
        #     mu_v=mu_v, logstd_v=logstd_v, actions_v=actions_v
        # ) * advantage_v.unsqueeze(dim=-1).detach()

        #entropy_v = self.calc_entropy(logstd_v=logstd_v)
        entropy_v = dist.entropy()

        loss_actor_v = -1.0 * reinforced_log_pi_action_v.mean()
        loss_entropy_v = -1.0 * entropy_v.mean()
        # loss_actor_v를 작아지도록 만듦 --> reinforced_log_pi_action_v.mean()가 커지도록 만듦
        # loss_entropy_v를 작아지도록 만듦 --> entropy_v가 커지도록 만듦

        self.actor_optimizer.zero_grad()
        (loss_actor_v + self.params.ENTROPY_LOSS_WEIGHT * loss_entropy_v).backward()
        self.actor_optimizer.step()

        self.buffer.clear()

        gradients = self.model.get_gradients_for_current_parameters()

        return gradients, loss_critic_v.item(), loss_actor_v.item() * -1.0

