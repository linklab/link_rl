import torch
import torch.nn.functional as F
from torch.distributions import Normal, Categorical

from codes.c_models.discrete_action.discrete_actor_critic_model import DiscreteActorCriticModel
from codes.d_agents.on_policy.a2c.a2c_agent import AgentA2C
from codes.d_agents.on_policy.stochastic_policy_action_selector import DiscreteCategoricalActionSelector
from codes.e_utils import rl_utils
from codes.e_utils.common_utils import show_info
from codes.e_utils.names import DeepLearningModelName, AgentMode


class AgentDiscreteA2C(AgentA2C):
    """
    """
    def __init__(
            self, worker_id, observation_shape, action_shape, action_n, params, device
    ):
        assert params.DEEP_LEARNING_MODEL in [
            DeepLearningModelName.DISCRETE_STOCHASTIC_ACTOR_CRITIC_MLP,
            DeepLearningModelName.DISCRETE_STOCHASTIC_ACTOR_CRITIC_CNN,
            DeepLearningModelName.DISCRETE_STOCHASTIC_ACTOR_CRITIC_RNN,
        ]
        super(AgentDiscreteA2C, self).__init__(worker_id, action_shape, params, device)

        self.__name__ = "AgentDiscreteA2C"
        self.action_n = action_n

        self.train_action_selector = DiscreteCategoricalActionSelector(params=params)
        self.test_and_play_action_selector = DiscreteCategoricalActionSelector(params=params)

        self.model = DiscreteActorCriticModel(
            worker_id=worker_id,
            observation_shape=observation_shape,
            action_n=action_n,
            params=params,
            device=device
        ).to(device)

        self.test_model = DiscreteActorCriticModel(
            worker_id=worker_id,
            observation_shape=observation_shape,
            action_n=action_n,
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

        # self.optimizer = rl_utils.get_optimizer(
        #     parameters=self.model.base.parameters(),
        #     learning_rate=self.params.LEARNING_RATE,
        #     params=params
        # )

    def __call__(self, state, agent_state=None):
        return self.discrete_call(state, agent_state)

    def on_train(self, step_idx, expected_model_version):
        batch = self.buffer.sample_all_for_on_policy(expected_model_version)

        # batch_states_v.shape: (32, 3)
        # batch_actions_v.shape: (32, 1)
        # batch_target_action_values_v.shape: (32,)
        batch_states_v, batch_actions_v, batch_target_action_values_v = self.unpack_batch_for_actor_critic(
            batch=batch, target_model=self.model, params=self.params
        )

        # batch_probs_v.shape: torch.Size([32, 2])
        # batch_value_v.shape: torch.Size([32, 1])
        batch_probs_v, batch_value_v = self.model.base.forward(batch_states_v)

        # Critic Optimization
        loss_critic_v = F.mse_loss(input=batch_value_v.squeeze(-1), target=batch_target_action_values_v.detach())

        self.critic_optimizer.zero_grad()
        loss_critic_v.backward(retain_graph=True)
        self.critic_optimizer.step()

        # advantage_v.shape: (32,)
        batch_advantage_v = batch_target_action_values_v - batch_value_v.squeeze(-1)

        dist = Categorical(probs=batch_probs_v)
        batch_reinforced_log_pi_action_v = dist.log_prob(value=batch_actions_v) * batch_advantage_v.unsqueeze(dim=-1).detach()
        batch_entropy_v = dist.entropy()

        loss_actor_v = -1.0 * batch_reinforced_log_pi_action_v.mean()
        loss_entropy_v = -1.0 * batch_entropy_v.mean()
        # loss_actor_v를 작아지도록 만듦 --> log_pi_v.mean()가 커지도록 만듦
        # loss_entropy_v를 작아지도록 만듦 --> entropy_v가 커지도록 만듦

        self.actor_optimizer.zero_grad()
        (loss_actor_v + self.params.ENTROPY_LOSS_WEIGHT * loss_entropy_v).backward()
        self.actor_optimizer.step()

        return None, loss_critic_v.item(), loss_actor_v.item() * -1.0
