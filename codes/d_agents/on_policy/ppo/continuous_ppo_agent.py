import torch
from torch.distributions import Normal, MultivariateNormal

from codes.a_config.parameters_general import StochasticActionSelectorType
from codes.c_models.continuous_action.continuous_stochastic_actor_critic_model import \
    StochasticContinuousActorCriticModel
from codes.d_agents.on_policy.stochastic_policy_action_selector import ContinuousNormalActionSelector, \
    SomeTimesBlowContinuousNormalActionSelector
from codes.d_agents.on_policy.ppo.ppo_agent import AgentPPO
from codes.e_utils import rl_utils
from codes.e_utils.common_utils import show_info
from codes.e_utils.names import DeepLearningModelName


class AgentContinuousPPO(AgentPPO):
    """
    """
    def __init__(
            self, worker_id, observation_shape, action_shape, num_outputs, action_min, action_max, params, device
    ):
        assert params.DEEP_LEARNING_MODEL in [
            DeepLearningModelName.CONTINUOUS_STOCHASTIC_ACTOR_CRITIC_MLP,
            DeepLearningModelName.CONTINUOUS_STOCHASTIC_ACTOR_CRITIC_CNN
        ]

        super(AgentContinuousPPO, self).__init__(
            worker_id=worker_id, action_shape=action_shape, params=params, device=device
        )
        self.num_outputs = num_outputs
        self.action_min = action_min
        self.action_max = action_max

        self.__name__ = "AgentContinuousPPO"

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

        # self.optimizer = rl_utils.get_actor_critic_optimizer(
        #     actor_parameters=self.model.base.actor_params,
        #     actor_learning_rate=self.params.ACTOR_LEARNING_RATE,
        #     critic_parameters=self.model.base.critic_params,
        #     critic_learning_rate=self.params.LEARNING_RATE,
        #     params=params
        # )

    def __call__(self, state, agent_state=None):
        return self.continuous_stochastic_call(state, agent_state)

    def on_train(self, step_idx, expected_model_version):
        trajectory = self.buffer.sample_all_for_on_policy(expected_model_version)

        # trajectory_states_v: (2049, 4)
        trajectory_states = [experience.state for experience in trajectory]
        trajectory_states_v = torch.FloatTensor(trajectory_states).to(self.device)

        # trajectory_actions_v: (2049,)
        trajectory_actions = [experience.action for experience in trajectory]
        trajectory_actions_v = torch.FloatTensor(trajectory_actions).to(self.device)

        # trajectory_mu_v.shape: (2049, 1)
        # trajectory_logstd_v.shpe: (2049, 1)
        # trajectory_values_v.shape: (2049, 1)
        trajectory_mu_v, trajectory_logstd_v, trajectory_values_v = self.model.base(trajectory_states_v)

        # trajectory_old_log_pi_action_v.shape: (2049, 1)
        trajectory_dist = Normal(loc=trajectory_mu_v, scale=torch.exp(trajectory_logstd_v))
        trajectory_old_log_pi_action_v = trajectory_dist.log_prob(value=trajectory_actions_v).detach()

        # 아래 변수는 전체 trajectory의 원소보다 1 적음
        with torch.no_grad():
            trajectory_advantage_v, trajectory_target_action_value_v = self.get_advantage_and_target_action_values(
                trajectory, trajectory_values_v, device=self.device
            )
            # normalize advantages
            trajectory_advantage_v = trajectory_advantage_v - torch.mean(trajectory_advantage_v)
            trajectory_advantage_v /= torch.std(trajectory_advantage_v) + 1e-7

        # drop last entry from the trajectory, an our adv and target action value calculated without it
        trajectory = trajectory[:-1]
        trajectory_states_v = trajectory_states_v[:-1]
        trajectory_actions_v = trajectory_actions_v[:-1]
        trajectory_old_log_pi_action_v = trajectory_old_log_pi_action_v[:-1]

        sum_loss_critic = 0.0
        sum_loss_actor = 0.0
        count_steps = 0

        for epoch in range(self.params.PPO_K_EPOCHS):
            for batch_offset in range(0, len(trajectory), self.params.PPO_TRAJECTORY_BATCH_SIZE):
                batch_l = batch_offset + self.params.PPO_TRAJECTORY_BATCH_SIZE

                batch_states_v = trajectory_states_v[batch_offset:batch_l]
                batch_actions_v = trajectory_actions_v[batch_offset:batch_l]
                batch_advantage_v = trajectory_advantage_v[batch_offset:batch_l].unsqueeze(-1)
                batch_target_action_value_v = trajectory_target_action_value_v[batch_offset:batch_l]
                batch_old_log_pi_action_v = trajectory_old_log_pi_action_v[batch_offset:batch_l]

                # batch_mu_v: (64, 1)
                # batch_var_v: (64, 1)
                # batch_values_v: (64, 1)
                batch_mu_v, batch_logstd_v, batch_values_v = self.model.base(batch_states_v)

                mean_batch_loss_critic_v = self.backward_and_step_for_critic(batch_values_v, batch_target_action_value_v)

                # actor training

                # METHOD 1
                # batch_action_variance.shape: (64, 1)
                # batch_covariance_matrix.shape: (64, 1, 1)
                # batch_log_pi_action_v.shape: (64, 1)
                # batch_entropy.shape: (64, 1)
                batch_dist = Normal(loc=batch_mu_v, scale=torch.exp(batch_logstd_v))
                batch_log_pi_action_v = batch_dist.log_prob(batch_actions_v)
                batch_entropy_v = batch_dist.entropy()

                # batch_advantage_v.shape: (64, 1)
                mean_batch_loss_actor_v = self.backward_and_step_for_actor(
                    batch_log_pi_action_v, batch_old_log_pi_action_v, batch_advantage_v, batch_entropy_v,
                )

                sum_loss_critic += mean_batch_loss_critic_v.item()
                sum_loss_actor += mean_batch_loss_actor_v.item()
                count_steps += 1

        #gradients = self.model.get_gradients_for_current_parameters()

        return None, sum_loss_critic / count_steps, (sum_loss_actor / count_steps) * -1.0