# https://awesomeopensource.com/project/nikhilbarhate99/PPO-PyTorch
import torch
import torch.nn.functional as F
from torch.distributions import Normal, MultivariateNormal

from codes.c_models.continuous_action.stochastic_continuous_actor_critic_model import StochasticContinuousActorCriticModel
from codes.d_agents.on_policy.a2c.a2c_agent import AgentA2C
from codes.d_agents.on_policy.on_policy_action_selector import ContinuousNormalActionSelector
from codes.e_utils import rl_utils
from codes.e_utils.common_utils import show_tensor_info
from codes.e_utils.names import DeepLearningModelName, AgentMode


class AgentContinuousA2C(AgentA2C):
    """
    """
    def __init__(
            self, worker_id, input_shape, action_shape, num_outputs, action_min, action_max, params, device
    ):
        assert params.DEEP_LEARNING_MODEL in [
            DeepLearningModelName.STOCHASTIC_CONTINUOUS_ACTOR_CRITIC_MLP
        ]
        super(AgentContinuousA2C, self).__init__(worker_id, action_shape, action_min, action_max, params, device)

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

    def __call__(self, states, agent_states=None):
        return self.continuous_stochastic_call(states, agent_states)

    def on_train(self, step_idx, expected_model_version):
        batch = self.buffer.sample_all_for_on_policy(expected_model_version)

        # states_v.shape: (32, 3)
        # actions_v.shape: (32, 1)
        # target_action_values_v.shape: (32,)
        batch_states_v, batch_actions_v, batch_target_action_values_v = self.unpack_batch_for_actor_critic(batch, self.model, self.params)

        # mu_v.shape: (32, 1)
        # var_v.shape: (32, 1)
        # value_v.shape; (32, 1)
        batch_mu_v, batch_logstd_v, batch_values_v = self.model.base(batch_states_v)

        # Critic Optimization
        loss_critic_v = F.mse_loss(input=batch_values_v.squeeze(-1), target=batch_target_action_values_v.detach())

        self.critic_optimizer.zero_grad()
        loss_critic_v.backward(retain_graph=True)
        self.critic_optimizer.step()

        # Actor Optimization
        # advantage_v.shape: (32,)
        batch_advantage_v = batch_target_action_values_v - batch_values_v.squeeze(-1)

        # reinforced_log_pi_action_v.shape: (32,)
        # batch_action_variance.shape: (32, 1)
        # batch_covariance_matrix.shape: (32, 1, 1)
        # batch_dist.log_prob(value=actions_v).shape: (32,)
        # batch_advantage_v.detach().shape: (32,)
        # batch_reinforced_log_pi_action_v.shape: (32,)
        # batch_action_variance = self.model.base.actor.action_variance.expand_as(mu_v)
        # batch_covariance_matrix = torch.diag_embed(batch_action_variance).to(self.device)
        # batch_dist = MultivariateNormal(loc=mu_v, covariance_matrix=batch_covariance_matrix)
        # batch_reinforced_log_pi_action_v = batch_dist.log_prob(value=actions_v) * batch_advantage_v.detach()
        # batch_entropy_v = batch_dist.entropy()

        dist = Normal(loc=batch_mu_v, scale=batch_logstd_v)

        # dist.log_prob(value=batch_actions_v).shape: (32, 1)
        # batch_entropy_v.shape: (32, 1)
        batch_reinforced_log_pi_action_v = dist.log_prob(value=batch_actions_v) * batch_advantage_v.unsqueeze(dim=-1).detach()
        batch_entropy_v = dist.entropy()

        # reinforced_log_pi_action_v = self.calc_logprob(
        #     mu_v=mu_v, logstd_v=logstd_v, actions_v=actions_v
        # ) * advantage_v.unsqueeze(dim=-1).detach()
        #entropy_v = self.calc_entropy(logstd_v=logstd_v)

        loss_actor_v = -1.0 * batch_reinforced_log_pi_action_v.mean()
        loss_entropy_v = -1.0 * batch_entropy_v.mean()
        # loss_actor_v를 작아지도록 만듦 --> reinforced_log_pi_action_v.mean()가 커지도록 만듦
        # loss_entropy_v를 작아지도록 만듦 --> entropy_v가 커지도록 만듦

        self.actor_optimizer.zero_grad()
        (loss_actor_v + self.params.ENTROPY_LOSS_WEIGHT * loss_entropy_v).backward()
        self.actor_optimizer.step()

        # gradients = self.model.get_gradients_for_current_parameters()

        return None, loss_critic_v.item(), loss_actor_v.item() * -1.0
