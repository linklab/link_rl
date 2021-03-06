# https://colab.research.google.com/github/nikhilbarhate99/PPO-PyTorch/blob/master/PPO_colab.ipynb#scrollTo=UT6VUBg-F8Zm
import torch
import torch.nn.functional as F
import torch.nn.utils as nn_utils

from codes.d_agents.on_policy.on_policy_agent import OnPolicyAgent
from codes.e_utils import replay_buffer
from codes.e_utils.common_utils import float32_preprocessor


class AgentPPO(OnPolicyAgent):
    """
    """
    def __init__(self, worker_id, action_shape, params, device):
        assert params.N_STEP == 1  # GAE will consider various N_STEPs

        super(AgentPPO, self).__init__(worker_id=worker_id, action_shape=action_shape, params=params, device=device)

        self.train_action_selector = None
        self.test_and_play_action_selector = None
        self.model = None
        self.critic_optimizer = None
        self.actor_optimizer = None
        self.optimizer = None
        self.buffer = replay_buffer.ExperienceReplayBuffer(
            experience_source=None, buffer_size=self.params.PPO_TRAJECTORY_SIZE
        )

    def __call__(self, state, critics=None):
        raise NotImplementedError

    def on_train(self, step_idx, expected_model_version):
        raise NotImplementedError

    def get_advantage_and_target_action_values(self, trajectory, values_v, device):
        """
        By trajectory calculate advantage and 1-step target action value
        :param trajectory: trajectory list
        :return: tuple with advantage numpy array and reference values
        """
        values = values_v.squeeze().data.cpu().numpy()

        # generalized advantage estimator: smoothed version of the advantage
        last_gae = 0.0
        result_advantages = []
        result_target_action_values = []
        for value, next_value, exp in zip(reversed(values[:-1]), reversed(values[1:]), reversed(trajectory[:-1])):
            if exp.done:
                delta = exp.reward - value
                last_gae = delta
            else:
                delta = exp.reward + self.params.GAMMA * next_value - value
                last_gae = delta + self.params.GAMMA * self.params.PPO_GAE_LAMBDA * last_gae

            result_advantages.append(last_gae)
            result_target_action_values.append(last_gae + value)

        advantage_v = float32_preprocessor(list(reversed(result_advantages)))
        target_action_value_v = float32_preprocessor(list(reversed(result_target_action_values)))
        return advantage_v.to(device), target_action_value_v.to(device)

    def backward_and_step_for_critic(self, batch_values_v, batch_target_action_value_v):
        # critic training
        # batch_values_v.squeeze(-1) : (64,)
        # batch_target_action_value_v : (64,)
        self.critic_optimizer.zero_grad()
        batch_loss_critic_v = F.mse_loss(batch_values_v.squeeze(-1), batch_target_action_value_v.detach())
        batch_loss_critic_v.backward(retain_graph=True)
        nn_utils.clip_grad_norm_(self.model.base.critic_params, self.params.CLIP_GRAD)
        self.critic_optimizer.step()
        return batch_loss_critic_v

    def backward_and_step_for_actor(
            self, batch_log_pi_action_v, batch_old_log_pi_action_v, batch_advantage_v, batch_entropy_v
    ):
        """
            # batch_log_pi_action_v.shape: (64, 1)
            # batch_old_log_pi_action_v.shape: (64, 1)
            # batch_advantage_v.shape: (64, 1)
            # batch_entropy_v.shape: (64, 1)
        """
        self.actor_optimizer.zero_grad()

        # batch_old_log_pi_action_v: (64, 1)
        # batch_ratio_v: (64, 1)
        batch_ratio_v = torch.exp(
            batch_log_pi_action_v - torch.maximum(
                batch_old_log_pi_action_v, torch.ones_like(batch_old_log_pi_action_v) * 1.0e-7
            )
        )

        # batch_advantage_1_v.shape: (64, 1)
        # batch_surrogate_2_v.shape: (64, 1)
        batch_surrogate_1_v = batch_advantage_v * batch_ratio_v
        batch_surrogate_2_v = batch_advantage_v * torch.clamp(
            batch_ratio_v, min=1.0 - self.params.PPO_EPSILON_CLIP, max=1.0 + self.params.PPO_EPSILON_CLIP
        )

        # mean_batch_loss_actor_v.shape: (1,)
        # mean_batch_loss_entropy_v.shape: (1,)
        mean_batch_loss_actor_v = -1.0 * torch.min(batch_surrogate_1_v, batch_surrogate_2_v).mean()
        mean_batch_loss_entropy_v = -1.0 * batch_entropy_v.mean()

        (mean_batch_loss_actor_v + self.params.ENTROPY_LOSS_WEIGHT * mean_batch_loss_entropy_v).backward()
        #nn_utils.clip_grad_norm(self.model.base.actor_params + self.model.base.critic_params, max_norm=2.0)
        nn_utils.clip_grad_norm_(self.model.base.actor_params, self.params.CLIP_GRAD)

        self.actor_optimizer.step()

        # if batch_advantage_v[0].item() < 0 and batch_ratio_v[0].item() > 1.0:
        #     print("ratio: {0:5.3f}, adv: {1:5.3f}, "
        #           "batch_surrogate_1_v: {2:5.3f}, batch_surrogate_2_v: {3:5.3f}, min: {4:5.3f}".format(
        #             batch_ratio_v[0].item(), batch_advantage_v[0].item(),
        #             batch_surrogate_1_v[0].item(), batch_surrogate_2_v[0].item(),
        #             min(batch_surrogate_1_v[0], batch_surrogate_2_v[0]).item()
        #         ), "*" if batch_surrogate_1_v[0].item() < batch_surrogate_2_v[0].item() else ""
        #     )
        #
        # if batch_advantage_v[1].item() < 0 and batch_ratio_v[1].item() > 1.0:
        #     print("ratio: {0:5.3f}, adv: {1:5.3f}, "
        #           "batch_surrogate_1_v: {2:5.3f}, batch_surrogate_2_v: {3:5.3f}, min: {4:5.3f}".format(
        #             batch_ratio_v[1].item(), batch_advantage_v[1].item(),
        #             batch_surrogate_1_v[1].item(), batch_surrogate_2_v[1].item(),
        #             min(batch_surrogate_1_v[1], batch_surrogate_2_v[1]).item()
        #         ), "*" if batch_surrogate_1_v[1].item() < batch_surrogate_2_v[1].item() else ""
        #     )

        return mean_batch_loss_actor_v