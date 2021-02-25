import torch
import torch.nn.functional as F
from torch.distributions import MultivariateNormal, Normal
import torch.nn.utils as nn_utils

from codes.d_agents.a0_base_agent import BaseAgent, float32_preprocessor
from codes.d_agents.on_policy.on_policy_agent import OnPolicyAgent
from codes.e_utils import rl_utils, replay_buffer
from codes.e_utils.actions import ContinuousNormalActionSelector
from codes.e_utils.names import DeepLearningModelName, AgentMode


class AgentContinuousPPO(OnPolicyAgent):
    """
    """
    def __init__(
            self, worker_id, input_shape, num_outputs, action_min, action_max, params, device
    ):
        assert params.DEEP_LEARNING_MODEL in [
            DeepLearningModelName.STOCHASTIC_CONTINUOUS_ACTOR_CRITIC_MLP,
            DeepLearningModelName.STOCHASTIC_CONTINUOUS_ACTOR_CRITIC_CNN
        ]
        assert params.N_STEP == 1  # GAE will consider various N_STEPs

        super(AgentContinuousPPO, self).__init__(params=params, device=device)

        self.__name__ = "AgentContinuousPPO"

        self.train_action_selector = ContinuousNormalActionSelector()
        self.test_and_play_action_selector = ContinuousNormalActionSelector()

        self.worker_id = worker_id
        self.action_min = action_min
        self.action_max = action_max

        self.model = rl_utils.get_rl_model(
            worker_id=worker_id, input_shape=input_shape, num_outputs=num_outputs, params=params, device=self.device
        )

        self.optimizer = rl_utils.get_optimizer(
            parameters=self.model.base.parameters(),
            learning_rate=self.params.LEARNING_RATE,
            params=params
        )

        self.buffer = replay_buffer.ExperienceReplayBuffer(
            experience_source=None, buffer_size=self.params.PPO_TRAJECTORY_SIZE
        )

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
        trajectory = self.buffer.sample(batch_size=None)

        trajectory_states = [experience.state for experience in trajectory]
        trajectory_states_v = torch.FloatTensor(trajectory_states).to(self.device)

        trajectory_actions = [experience.action for experience in trajectory]
        trajectory_actions_v = torch.FloatTensor(trajectory_actions).to(self.device)

        trajectory_mu_v, trajectory_var_v, trajectory_values_v = self.model.base.forward(trajectory_states_v)

        # trajectory_var_v = self.model.base.actor.var.expand_as(trajectory_mu_v)
        # trajectory_covariance_matrix = torch.diag_embed(trajectory_var_v).to(self.device)
        # trajectory_dist = MultivariateNormal(loc=trajectory_mu_v, covariance_matrix=trajectory_covariance_matrix)

        trajectory_dist = Normal(loc=trajectory_mu_v, scale=torch.sqrt(trajectory_var_v))
        trajectory_old_log_pi_action_v = trajectory_dist.log_prob(trajectory_actions_v)

        # 아래 변수는 전체 trajectory의 원소보다 1 적음
        with torch.no_grad():
            trajectory_advantage_v, trajectory_target_action_value_v = self.get_advantage_and_target_action_values(
                trajectory, trajectory_values_v, device=self.device
            )
            # normalize advantages
            trajectory_advantage_v = trajectory_advantage_v - trajectory_advantage_v.mean()
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

                batch_mu_v, batch_var_v, batch_values_v = self.model(batch_states_v)

                # critic training
                loss_critic_v = F.smooth_l1_loss(batch_values_v.squeeze(-1), batch_target_action_value_v.detach())

                # actor training
                batch_dist = Normal(loc=batch_mu_v, scale=torch.sqrt(batch_var_v))

                batch_log_pi_action_v = batch_dist.log_prob(batch_actions_v)
                batch_dist_entropy_v = batch_dist.entropy()

                batch_ratio_v = torch.exp(batch_log_pi_action_v - batch_old_log_pi_action_v)

                batch_surrogate_1_v = batch_advantage_v * batch_ratio_v
                batch_surrogate_2_v = batch_advantage_v * torch.clamp(
                    batch_ratio_v, min=1.0 - self.params.PPO_EPSILON_CLIP, max=1.0 + self.params.PPO_EPSILON_CLIP
                )
                loss_actor_v = -1.0 * torch.min(batch_surrogate_1_v, batch_surrogate_2_v).mean()
                loss_entropy_v = -1.0 * batch_dist_entropy_v.mean()

                loss_v = loss_actor_v + \
                         self.params.CRITIC_LOSS_WEIGHT * loss_critic_v + \
                         self.params.ENTROPY_LOSS_WEIGHT * loss_entropy_v

                self.optimizer.zero_grad()
                loss_v.backward()
                nn_utils.clip_grad_norm_(self.model.base.parameters(), self.params.CLIP_GRAD)
                self.optimizer.step()

                sum_loss_critic += loss_critic_v.item()
                sum_loss_actor += loss_actor_v.item()
                count_steps += 1

        gradients = self.model.get_gradients_for_current_parameters()

        return gradients, sum_loss_critic / count_steps, (sum_loss_actor / count_steps) * -1.0