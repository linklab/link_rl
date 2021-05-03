import torch
import torch.nn.functional as F

from codes.d_agents.on_policy.on_policy_agent import OnPolicyAgent
from codes.e_utils import replay_buffer


class AgentPPO(OnPolicyAgent):
    """
    """
    def __init__(self, worker_id, params, action_shape, device):
        assert params.N_STEP == 1  # GAE will consider various N_STEPs

        super(AgentPPO, self).__init__(worker_id=worker_id, params=params, action_shape=action_shape, device=device)

        self.train_action_selector = None
        self.test_and_play_action_selector = None
        self.model = None
        self.critic_optimizer = None
        self.actor_optimizer = None
        self.buffer = replay_buffer.ExperienceReplayBuffer(
            experience_source=None, buffer_size=self.params.PPO_TRAJECTORY_SIZE
        )

    def __call__(self, states, critics=None):
        raise NotImplementedError

    def train(self, step_idx):
        raise NotImplementedError

    def backward_and_step_for_critic(self, batch_values_v, batch_target_action_value_v):
        # critic training
        # batch_values_v.squeeze(-1) : (64,)
        # batch_target_action_value_v : (64,)
        self.critic_optimizer.zero_grad()
        batch_loss_critic_v = F.mse_loss(batch_values_v.squeeze(-1), batch_target_action_value_v.detach())
        batch_loss_critic_v.backward()
        self.critic_optimizer.step()
        return batch_loss_critic_v

    def backward_and_step_for_actor(
            self, batch_log_pi_action_v, batch_old_log_pi_action_v, batch_advantage_v, batch_loss_entropy_v
    ):
        self.actor_optimizer.zero_grad()

        # batch_old_log_pi_action_v: (64, 1)
        # batch_ratio_v: (64, 1)
        batch_ratio_v = torch.exp(batch_log_pi_action_v - batch_old_log_pi_action_v)

        # batch_advantage_v: (64, 1)
        batch_surrogate_1_v = batch_advantage_v * batch_ratio_v
        batch_surrogate_2_v = batch_advantage_v * torch.clamp(
            batch_ratio_v, min=1.0 - self.params.PPO_EPSILON_CLIP, max=1.0 + self.params.PPO_EPSILON_CLIP
        )
        batch_loss_actor_v = -1.0 * torch.min(batch_surrogate_1_v, batch_surrogate_2_v).mean()

        (batch_loss_actor_v + self.params.ENTROPY_LOSS_WEIGHT * batch_loss_entropy_v).backward()
        #nn_utils.clip_grad_norm_(self.model.base.parameters(), self.params.CLIP_GRAD)
        self.actor_optimizer.step()
        return batch_loss_actor_v