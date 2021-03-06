# https://awesomeopensource.com/project/nikhilbarhate99/PPO-PyTorch
import torch
import torch.nn.functional as F
from torch.distributions import Normal, MultivariateNormal

from codes.a_config.parameters_general import StochasticActionSelectorType
from codes.c_models.base_model import RNNModel
from codes.c_models.continuous_action.continuous_stochastic_actor_critic_model import StochasticContinuousActorCriticModel
from codes.d_agents.on_policy.a2c.a2c_agent import AgentA2C
from codes.d_agents.on_policy.stochastic_policy_action_selector import ContinuousNormalActionSelector, \
    SomeTimesBlowContinuousNormalActionSelector
from codes.e_utils import rl_utils
from codes.e_utils.common_utils import show_info
from codes.e_utils.names import DeepLearningModelName


class AgentContinuousA2C(AgentA2C):
    """
    """
    def __init__(
            self, worker_id, observation_shape, action_shape, num_outputs, action_min, action_max, params, device
    ):
        assert params.DEEP_LEARNING_MODEL in [
            DeepLearningModelName.CONTINUOUS_STOCHASTIC_ACTOR_CRITIC_MLP
        ]
        super(AgentContinuousA2C, self).__init__(worker_id, action_shape, params, device)
        self.__name__ = "AgentContinuousA2C"

        self.num_outputs = num_outputs
        self.action_min = action_min
        self.action_max = action_max

        if params.TYPE_OF_STOCHASTIC_ACTION_SELECTOR == StochasticActionSelectorType.BASIC_ACTION_SELECTOR:
            self.train_action_selector = ContinuousNormalActionSelector(params=params)
        elif params.TYPE_OF_STOCHASTIC_ACTION_SELECTOR == StochasticActionSelectorType.SOMETIMES_BLOW_ACTION_SELECTOR:
            self.train_action_selector = SomeTimesBlowContinuousNormalActionSelector(
                min_blowing_action=-5.0, max_blowing_action=5.0, params=self.params,
            )
        else:
            raise ValueError()

        self.test_and_play_action_selector = ContinuousNormalActionSelector(params=params)

        self.model = StochasticContinuousActorCriticModel(
            worker_id=worker_id,
            observation_shape=observation_shape,
            num_outputs=num_outputs,
            params=params,
            device=device
        ).to(device)

        self.test_model = StochasticContinuousActorCriticModel(
            worker_id=worker_id,
            observation_shape=observation_shape,
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

    def __call__(self, state, agent_state=None):
        return self.continuous_stochastic_call(state, agent_state)

    def on_train(self, step_idx, expected_model_version):
        batch = self.buffer.sample_all_for_on_policy(expected_model_version)

        # states_v.shape: (32, 3)
        # actions_v.shape: (32, 1)
        # target_action_values_v.shape: (32,)
        if isinstance(self.model, RNNModel):
            batch_states_v, batch_actions_v, batch_target_action_values_v, batch_actor_hidden_states_v, \
            batch_critic_hidden_states_v = self.unpack_batch_for_actor_critic(
                batch=batch, target_model=self.model, params=self.params
            )
        else:
            batch_states_v, batch_actions_v, batch_target_action_values_v = self.unpack_batch_for_actor_critic(
                batch=batch, target_model=self.model, params=self.params
            )
            batch_actor_hidden_states_v = batch_critic_hidden_states_v = None

        # mu_v.shape: (32, 1)
        # var_v.shape: (32, 1)
        # value_v.shape; (32, 1)
        batch_mu_v, batch_logstd_v, new_batch_actor_hidden_states = self.model.forward_actor(
            batch_states_v, batch_actor_hidden_states_v
        )

        batch_values_v, new_batch_critic_hidden_states = self.model.forward_critic(
            batch_states_v, batch_critic_hidden_states_v
        )

        # Critic Optimization
        loss_critic_v = F.mse_loss(input=batch_values_v.squeeze(-1), target=batch_target_action_values_v.detach())

        self.critic_optimizer.zero_grad()
        loss_critic_v.backward(retain_graph=True)
        self.critic_optimizer.step()

        # Actor Optimization
        # batch_advantage_v.shape: (32,)
        batch_advantage_v = batch_target_action_values_v - batch_values_v.squeeze(-1)

        dist = Normal(loc=batch_mu_v, scale=torch.exp(batch_logstd_v))

        # dist.log_prob(value=batch_actions_v).shape: (32, 1)
        # batch_reinforced_log_pi_action_v.shape: (32, 1)
        # batch_entropy_v.shape: (32, 1)
        batch_reinforced_log_pi_action_v = dist.log_prob(value=batch_actions_v) * batch_advantage_v.unsqueeze(dim=-1).detach()
        batch_entropy_v = dist.entropy()

        loss_actor_v = -1.0 * batch_reinforced_log_pi_action_v.mean()
        loss_entropy_v = -1.0 * batch_entropy_v.mean()
        # loss_actor_v??? ??????????????? ?????? --> reinforced_log_pi_action_v.mean()??? ???????????? ??????
        # loss_entropy_v??? ??????????????? ?????? --> entropy_v??? ???????????? ??????

        self.actor_optimizer.zero_grad()
        (loss_actor_v + self.params.ENTROPY_LOSS_WEIGHT * loss_entropy_v).backward()
        self.actor_optimizer.step()

        # gradients = self.model.get_gradients_for_current_parameters()

        return None, loss_critic_v.item(), loss_actor_v.item() * -1.0
