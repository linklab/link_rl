import torch
import torch.nn.functional as F
from torch.distributions import Normal, MultivariateNormal
import torch.nn.utils as nn_utils

from codes.d_agents.a0_base_agent import BaseAgent, float32_preprocessor
from codes.d_agents.on_policy.on_policy_agent import OnPolicyAgent
from codes.e_utils import rl_utils, replay_buffer
from codes.e_utils.actions import ContinuousNormalActionSelector
from codes.e_utils.names import DeepLearningModelName, AgentMode


class AgentContinuousA2C(OnPolicyAgent):
    """
    """
    def __init__(
            self, worker_id, input_shape, num_outputs, action_min, action_max, params, device="cpu"
    ):
        assert params.DEEP_LEARNING_MODEL == DeepLearningModelName.STOCHASTIC_CONTINUOUS_ACTOR_CRITIC_MLP

        super(AgentContinuousA2C, self).__init__(params, device)
        self.__name__ = "AgentContinuousA2C"
        self.train_action_selector = ContinuousNormalActionSelector()
        self.test_and_play_action_selector = ContinuousNormalActionSelector()
        self.worker_id = worker_id
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

        # self.optimizer = rl_utils.get_optimizer(
        #     parameters=self.model.base.parameters(),
        #     learning_rate=self.params.LEARNING_RATE,
        #     params=params
        # )

        self.buffer = replay_buffer.ExperienceReplayBuffer(experience_source=None, buffer_size=self.params.BATCH_SIZE)

    def __call__(self, states, critics=None):
        if not isinstance(states, torch.FloatTensor):
            states = float32_preprocessor(states).to(self.device)

        mu_v, var_v = self.model.base.actor(states)

        if self.agent_mode == AgentMode.TRAIN:
            actions = self.train_action_selector(mu_v, var_v, self.action_min, self.action_max)
        else:
            actions = self.test_and_play_action_selector(mu_v, var_v, self.action_min, self.action_max)

        critics = torch.zeros(size=mu_v.size())

        return actions, critics

    def train(self, step_idx):
        batch = self.buffer.sample(batch_size=None)

        # states_v.shape: (32, 3)
        # actions_v.shape: (32, 1)
        # target_action_values_v.shape: (32,)

        states_v, actions_v, target_action_values_v = self.unpack_batch_for_actor_critic(batch, self.model, self.params)

        # mu_v.shape: (32, 1)
        # var_v.shape: (32, 1)
        # value_v.shape; (32, 1)
        mu_v, var_v, value_v = self.model(states_v)

        # Critic Optimization
        loss_critic_v = F.mse_loss(input=value_v.squeeze(-1), target=target_action_values_v.detach())

        self.critic_optimizer.zero_grad()
        loss_critic_v.backward()
        nn_utils.clip_grad_norm_(self.model.base.critic.parameters(), self.params.CLIP_GRAD)
        self.critic_optimizer.step()

        # Actor Optimization
        # advantage_v.shape: (32,)
        advantage_v = target_action_values_v - value_v.squeeze(-1)

        # covariance_matrix = torch.diag_embed(var_v).to(self.device)
        # dist = MultivariateNormal(loc=mu_v, covariance_matrix=covariance_matrix)
        # log_pi_action_v = advantage_v * dist.log_prob(actions_v).unsqueeze(-1)
        dist = Normal(loc=mu_v, scale=torch.sqrt(var_v))
        reinforced_log_pi_action_v = advantage_v.detach() * dist.log_prob(actions_v).squeeze(-1)

        #print(advantage_v.size(), dist.log_prob(actions_v).squeeze(-1).size(), reinforced_log_pi_action_v.size())

        loss_actor_v = -1.0 * reinforced_log_pi_action_v.mean()
        loss_entropy_v = -1.0 * dist.entropy().mean()

        # loss_actor_v를 작아지도록 만듦 --> log_pi_v.mean()가 커지도록 만듦
        # loss_entropy_v를 작아지도록 만듦 --> entropy_v가 커지도록 만듦
        # print(loss_critic_v, loss_actor_v, loss_entropy_v)
        # loss_v = loss_actor_v + \
        #          self.params.CRITIC_LOSS_WEIGHT * loss_critic_v + \
        #          self.params.ENTROPY_LOSS_WEIGHT * loss_entropy_v

        self.actor_optimizer.zero_grad()
        (loss_actor_v + self.params.ENTROPY_LOSS_WEIGHT * loss_entropy_v).backward()
        nn_utils.clip_grad_norm_(self.model.base.actor.parameters(), self.params.CLIP_GRAD)
        self.actor_optimizer.step()

        gradients = self.model.get_gradients_for_current_parameters()

        self.buffer.clear()

        return gradients, loss_critic_v.item(), loss_actor_v.item() * -1.0