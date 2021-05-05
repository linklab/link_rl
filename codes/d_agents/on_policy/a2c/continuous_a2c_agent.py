import torch
import torch.nn.functional as F
from torch.distributions import Normal

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

        for name, param in self.model.base.named_parameters():
            print(name, param.shape, "!!!!!!!!!!!!!!!!!!!!!!!!")

        self.optimizer = rl_utils.get_optimizer(
            parameters=self.model.base.parameters(),
            learning_rate=self.params.LEARNING_RATE,
            params=params
        )

    def __call__(self, states, critics=None):
        return self.continuous_call(states, critics)

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

        # Actor Optimization
        # advantage_v.shape: (32,)
        advantage_v = target_action_values_v - value_v.squeeze(-1)

        # covariance_matrix = torch.diag_embed(var_v).to(self.device)
        # dist = MultivariateNormal(loc=mu_v, covariance_matrix=covariance_matrix)
        # log_pi_action_v = advantage_v * dist.log_prob(actions_v).unsqueeze(-1)
        dist = Normal(loc=mu_v, scale=logstd_v)

        reinforced_log_pi_action_v = advantage_v.unsqueeze(dim=-1).detach() * dist.log_prob(actions_v)

        #print(reinforced_log_pi_action_v.shape, reinforced_log_pi_action_v.mean().shape, dist.entropy().shape, dist.entropy().mean().shape)

        loss_actor_v = -1.0 * reinforced_log_pi_action_v.mean()
        loss_entropy_v = -1.0 * dist.entropy().mean()
        # loss_actor_v를 작아지도록 만듦 --> log_pi_v.mean()가 커지도록 만듦
        # loss_entropy_v를 작아지도록 만듦 --> entropy_v가 커지도록 만듦

        return self.backward_and_step(loss_critic_v, loss_entropy_v, loss_actor_v)
