import torch
from torch.distributions import Normal

from codes.c_models.continuous_action.stochastic_continuous_actor_critic_model import \
    StochasticContinuousActorCriticModel
from codes.d_agents.on_policy.on_policy_action_selector import ContinuousNormalActionSelector
from codes.d_agents.on_policy.ppo.ppo_agent import AgentPPO
from codes.e_utils import rl_utils
from codes.e_utils.names import DeepLearningModelName


class AgentContinuousPPO(AgentPPO):
    """
    """
    def __init__(
            self, worker_id, input_shape, action_shape, num_outputs, params, device
    ):
        assert params.DEEP_LEARNING_MODEL in [
            DeepLearningModelName.STOCHASTIC_CONTINUOUS_ACTOR_CRITIC_MLP,
            DeepLearningModelName.STOCHASTIC_CONTINUOUS_ACTOR_CRITIC_CNN
        ]

        super(AgentContinuousPPO, self).__init__(
            worker_id=worker_id, params=params, action_shape=action_shape, device=device
        )
        self.__name__ = "AgentContinuousPPO"

        self.train_action_selector = ContinuousNormalActionSelector()
        self.test_and_play_action_selector = ContinuousNormalActionSelector()

        self.model = StochasticContinuousActorCriticModel(
            worker_id=worker_id,
            input_shape=input_shape,
            num_outputs=num_outputs,
            params=params,
            device=device
        ).to(device)

        # self.optimizer = rl_utils.get_optimizer(
        #     parameters=self.model.base.parameters(),
        #     learning_rate=self.params.LEARNING_RATE,
        #     params=params
        # )

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

    def __call__(self, states, critics=None):
        return self.continuous_call(states, critics)

    def train(self, step_idx):
        trajectory = self.buffer.sample(batch_size=None)

        # trajectory_states_v: (2049, 4)
        trajectory_states = [experience.state for experience in trajectory]
        trajectory_states_v = torch.FloatTensor(trajectory_states).to(self.device)

        # trajectory_actions_v: (2049,)
        trajectory_actions = [experience.action for experience in trajectory]
        trajectory_actions_v = torch.FloatTensor(trajectory_actions).to(self.device)

        # trajectory_probs_v: (2049, 2)
        # trajectory_values_v: (2849, 1)
        trajectory_mu_v, trajectory_logstd_v, trajectory_values_v = self.model.base.forward(trajectory_states_v)

        # trajectory_var_v = self.model.base.actor.var.expand_as(trajectory_mu_v)
        # trajectory_covariance_matrix = torch.diag_embed(trajectory_var_v).to(self.device)
        # trajectory_dist = MultivariateNormal(loc=trajectory_mu_v, covariance_matrix=trajectory_covariance_matrix)

        # trajectory_dist = Normal(loc=trajectory_mu_v, scale=torch.sqrt(trajectory_var_v))
        # trajectory_old_log_pi_action_v = trajectory_dist.log_prob(trajectory_actions_v)

        trajectory_old_log_pi_action_v = self.calc_logprob(
            mu_v=trajectory_mu_v, logstd_v=trajectory_logstd_v, actions_v=trajectory_actions_v
        )

        # 아래 변수는 전체 trajectory의 원소보다 1 적음
        with torch.no_grad():
            trajectory_advantage_v, trajectory_target_action_value_v = self.get_advantage_and_target_action_values(
                trajectory, trajectory_values_v, device=self.device
            )
            # normalize advantages
            trajectory_advantage_v = trajectory_advantage_v - torch.mean(trajectory_advantage_v)
            trajectory_advantage_v /= torch.std(trajectory_advantage_v) + 1e-5

        # drop last entry from the trajectory, an our adv and target action value calculated without it
        trajectory = trajectory[:-1]
        trajectory_states_v = trajectory_states_v[:-1]
        trajectory_actions_v = trajectory_actions_v[:-1]
        trajectory_old_log_pi_action_v = trajectory_old_log_pi_action_v[:-1].detach()

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
                batch_mu_v, batch_logstd_v, batch_values_v = self.model(batch_states_v)

                batch_loss_critic_v = self.backward_and_step_for_critic(batch_values_v, batch_target_action_value_v)

                # batch_dist = Normal(loc=batch_mu_v, scale=torch.sqrt(batch_var_v))
                # batch_log_pi_action_v = batch_dist.log_prob(batch_actions_v)

                # actor training
                # batch_actions_v: (64, 1)
                # batch_log_pi_action_v: (64, 1)
                batch_log_pi_action_v = self.calc_logprob(
                    mu_v=batch_mu_v, logstd_v=batch_logstd_v, actions_v=batch_actions_v
                )

                batch_dist_entropy_v = self.calc_entropy(logstd_v=batch_logstd_v)

                batch_loss_entropy_v = -1.0 * batch_dist_entropy_v.mean()

                batch_loss_actor_v = self.backward_and_step_for_actor(
                    batch_log_pi_action_v, batch_old_log_pi_action_v, batch_advantage_v, batch_loss_entropy_v
                )

                sum_loss_critic += batch_loss_critic_v.item()
                sum_loss_actor += batch_loss_actor_v.item()
                count_steps += 1

        gradients = self.model.get_gradients_for_current_parameters()

        return gradients, sum_loss_critic / count_steps, (sum_loss_actor / count_steps) * -1.0