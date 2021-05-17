# https://spinningup.openai.com/en/latest/algorithms/sac.html
# https://github.com/pranz24/pytorch-soft-actor-critic
#https://github.com/ku2482/soft-actor-critic.pytorch/blob/master/code/agent.py
import torch
import torch.nn.functional as F
import torch.nn.utils as nn_utils

from codes.c_models.continuous_action.soft_actor_critic_model import SoftActorCriticModel
from codes.d_agents.off_policy.off_policy_agent import OffPolicyAgent
from codes.d_agents.off_policy.sac.sac_action_selector import ContinuousNormalSACActionSelector
from codes.e_utils import rl_utils
from codes.e_utils.common_utils import grad_false
from codes.e_utils.names import DeepLearningModelName


class AgentSAC(OffPolicyAgent):
    """
    """
    def __init__(
            self, worker_id, input_shape, action_shape, num_outputs, params, device
    ):
        assert params.DEEP_LEARNING_MODEL == DeepLearningModelName.SOFT_ACTOR_CRITIC_MLP

        super(AgentSAC, self).__init__(worker_id, params, action_shape, device)
        self.__name__ = "AgentSAC"

        self.train_action_selector = ContinuousNormalSACActionSelector(params=params)
        self.test_and_play_action_selector = ContinuousNormalSACActionSelector(params=params)

        self.model = SoftActorCriticModel(
            worker_id=worker_id,
            input_shape=input_shape,
            num_outputs=num_outputs,
            params=params,
            device=device
        ).to(device)

        self.target_model = SoftActorCriticModel(
            worker_id=worker_id,
            input_shape=input_shape,
            num_outputs=num_outputs,
            params=params,
            device=device
        ).to(device)

        grad_false(self.target_model)

        self.test_model = SoftActorCriticModel(
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

        self.twinq_optimizer = rl_utils.get_optimizer(
            parameters=self.model.base.twinq_params,
            learning_rate=self.params.LEARNING_RATE,
            params=params
        )

        self.alpha = None

    def reset_alpha(self):
        if self.params.ENTROPY_TUNING:
            # Target entropy is -|A|.
            self.target_entropy = -torch.prod(torch.Tensor(self.action_shape).to(self.device)).item()

            print(self.target_entropy, "!")
            # We optimize log(alpha), instead of alpha.
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha = self.log_alpha.exp()
            self.alpha_optimizer = rl_utils.get_optimizer(
                parameters=[self.log_alpha],
                learning_rate=self.params.LEARNING_RATE,
                params=self.params
            )
        else:
            # fixed alpha
            self.alpha = torch.tensor(self.params.ALPHA).to(self.device)

    def __call__(self, states, critics=None):
        if self.alpha is None:
            self.reset_alpha()
        return self.continuous_stochastic_call(states, critics)

    def train(self, step_idx):
        if self.alpha is None:
            self.reset_alpha()

        if self.params.PER:
            batch, batch_indices, batch_weights = self.buffer.sample(self.params.BATCH_SIZE)
        else:
            batch = self.buffer.sample(self.params.BATCH_SIZE)
            batch_indices, batch_weights = None, None

        # print(batch)
        states_v, actions_v, target_values_v, target_action_values_v = self.unpack_batch_for_sac(batch)

        # train twinq
        self.twinq_optimizer.zero_grad()
        q1_v, q2_v = self.model.base.twinq(states_v, actions_v)
        q_loss_v = F.mse_loss(q1_v.squeeze(), target_action_values_v.detach(), reduction="none") + \
                   F.mse_loss(q2_v.squeeze(), target_action_values_v.detach(), reduction="none")
        q_loss_v = q_loss_v.mean()
        q_loss_v.backward(retain_graph=True)
        nn_utils.clip_grad_norm_(self.model.base.twinq.parameters(), self.params.CLIP_GRAD)
        self.twinq_optimizer.step()

        if self.params.PER_PROPORTIONAL or self.params.PER_RANK_BASED:
            self.buffer.update_priorities(batch_indices, q_loss_v.abs().detach().cpu().numpy())
            self.buffer.update_beta(step_idx)

        # train actor
        self.actor_optimizer.zero_grad()
        sampled_action_v, sampled_entropies_v = self.model.sample(states_v)
        q1_v, q2_v = self.model.base.twinq(states_v, sampled_action_v)

        # q1_v.shape: [128, 1]
        # q2_v.shape: [128, 1]
        # loss_actor_v = -1.0 * (torch.min(q1_v, q2_v).squeeze() - self.params.ALPHA * sampled_log_prob).mean()
        loss_actor_v = -1.0 * (torch.min(q1_v, q2_v).squeeze() + self.alpha * sampled_entropies_v).mean()
        loss_actor_v.backward(retain_graph=True)
        nn_utils.clip_grad_norm_(self.model.base.actor_params, self.params.CLIP_GRAD)
        self.actor_optimizer.step()

        self.target_model.twinq_alpha_sync(self.model, alpha=1 - self.params.TAU)

        if self.params.ENTROPY_TUNING:
            self.alpha_optimizer.zero_grad()
            # Intuitively, we increase alpha when entropy is less than target entropy, vice versa.
            entropy_loss = -1.0 * self.log_alpha * (self.target_entropy - sampled_entropies_v).mean().detach()
            entropy_loss.backward()
            nn_utils.clip_grad_norm_([self.log_alpha], self.params.CLIP_GRAD)
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp()

        # gradients = self.model.get_gradients_for_current_parameters()
        gradients = None

        return gradients, q_loss_v.item(), loss_actor_v.item() * -1.0

    def unpack_batch_for_sac(self, batch):
        # states_v.shape: [128, 3]
        # actions_v.shape: [128]
        # target_action_values_v.shape: [128]
        states_v, actions_v, target_action_values_v = self.unpack_batch_for_actor_critic(
            batch, self.target_model, self.params, sac=True, alpha=self.alpha
        )

        sampled_action_v, sampled_entropies_v = self.model.sample(states_v)
        q1_v, q2_v = self.model.base.twinq(states_v, sampled_action_v)

        # torch.min(q1_v, q2_v).squeeze().shape: [128]
        # sampled_log_prob.shape: [128]
        target_values_v = torch.min(q1_v, q2_v).squeeze() + self.alpha * sampled_entropies_v
        return states_v, actions_v, target_values_v, target_action_values_v
