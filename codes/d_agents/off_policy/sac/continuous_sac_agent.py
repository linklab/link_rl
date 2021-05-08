# https://spinningup.openai.com/en/latest/algorithms/sac.html
# https://github.com/pranz24/pytorch-soft-actor-critic
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Normal
import torch.nn.utils as nn_utils

from codes.c_models.continuous_action.soft_actor_critic_model import SoftActorCriticModel
from codes.d_agents.a0_base_agent import float32_preprocessor
from codes.d_agents.off_policy.off_policy_agent import OffPolicyAgent
from codes.d_agents.off_policy.sac.sac_action_selector import ContinuousNormalSACActionSelector
from codes.d_agents.on_policy.on_policy_agent import OnPolicyAgent
from codes.e_utils import rl_utils, replay_buffer
from codes.e_utils.names import DeepLearningModelName, AgentMode


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

    def __call__(self, states, critics=None):
        return self.continuous_stochastic_call(states, critics)

    def train(self, step_idx):
        if self.params.PER:
            batch, batch_indices, batch_weights = self.buffer.sample(self.params.BATCH_SIZE)
        else:
            batch = self.buffer.sample(self.params.BATCH_SIZE)
            batch_indices, batch_weights = None, None

        # print(batch)
        states_v, actions_v, target_action_values_v = self.unpack_batch_for_sac(batch)

        # train twinq
        self.twinq_optimizer.zero_grad()
        q1_v, q2_v = self.model.base.twinq(states_v, actions_v)
        q_loss_v = F.mse_loss(q1_v.squeeze(), target_action_values_v.detach(), reduction="none") + \
                   F.mse_loss(q2_v.squeeze(), target_action_values_v.detach(), reduction="none")
        q_loss_v = q_loss_v.mean()
        q_loss_v.backward()
        nn_utils.clip_grad_norm_(self.model.base.twinq.parameters(), self.params.CLIP_GRAD)
        self.twinq_optimizer.step()

        # # train critic
        # self.critic_optimizer.zero_grad()
        # val_v = self.model.base.critic(states_v)
        #
        # if self.params.PER:
        #     batch_l1_loss = F.smooth_l1_loss(val_v.squeeze(), target_values_v.detach(), reduction="none")
        #     batch_weights_v = torch.tensor(batch_weights)
        #     critic_loss_v = batch_weights_v * batch_l1_loss
        #
        #     self.buffer.update_priorities(batch_indices, batch_l1_loss.detach().cpu().numpy() + 1e-5)
        #     self.buffer.update_beta(step_idx)
        # else:
        #     # val_v.squeeze().shape: [128]
        #     # target_values_v.shape: [128]
        #     # critic_loss_v.shape: [128]
        #     critic_loss_v = F.mse_loss(val_v.squeeze(), target_values_v.detach(), reduction="none")
        #
        # loss_critic_v = critic_loss_v.mean()
        # loss_critic_v.backward()
        # nn_utils.clip_grad_norm_(self.model.base.critic.parameters(), self.params.CLIP_GRAD)
        # self.critic_optimizer.step()

        # train actor
        self.actor_optimizer.zero_grad()
        mu_v, logstd_v = self.model.base.actor(states_v)
        act_dist = Normal(loc=mu_v, scale=torch.exp(logstd_v))
        acts_v = act_dist.rsample()
        q1_v, q2_v = self.model.base.twinq(states_v, acts_v)

        # q1_v.shape: [128, 1]
        # q2_v.shape: [128, 1]
        loss_actor_v = -1.0 * torch.min(q1_v, q2_v).squeeze().mean()
        loss_actor_v.backward()
        nn_utils.clip_grad_norm_(self.model.base.actor_params, self.params.CLIP_GRAD)
        self.actor_optimizer.step()

        self.target_model.alpha_sync(self.model, alpha=1 - self.params.TAU)

        gradients = self.model.get_gradients_for_current_parameters()

        return gradients, q_loss_v.item(), loss_actor_v.item() * -1.0

    def unpack_batch_for_sac(self, batch):
        states, actions, rewards, not_done_idx, last_states = [], [], [], [], []

        for idx, exp in enumerate(batch):
            states.append(np.array(exp.state, copy=False))
            actions.append(exp.action)
            rewards.append(exp.reward)
            if exp.last_state is not None:
                not_done_idx.append(idx)
                last_states.append(np.array(exp.last_state, copy=False))

        states_v = float32_preprocessor(states).to(self.device)
        actions_v = float32_preprocessor(actions).to(self.device)
        target_action_values_np = np.array(rewards, dtype=np.float32)

        last_states_v = torch.FloatTensor(np.array(last_states, copy=False)).to(self.device)
        last_action_v, last_log_prob = self.model.sample(last_states_v)

        if not_done_idx:
            last_q_values_v_1, last_q_values_v_2 = self.target_model.base.twinq.forward(last_states_v, last_action_v)
            last_min_q_values_v = torch.min(last_q_values_v_1, last_q_values_v_2)
            last_values_np = last_min_q_values_v.data.cpu().numpy()[:, 0] * (self.params.GAMMA ** self.params.N_STEP)
            target_action_values_np[not_done_idx] += last_values_np

        target_action_values_v = float32_preprocessor(target_action_values_np).to(self.device) - self.params.ALPHA * last_log_prob

        # mu_v, logstd_v = self.model.base.actor(states_v)
        # act_dist = Normal(mu_v, torch.exp(logstd_v))
        # acts_v = act_dist.sample()
        # q1_v, q2_v = self.model.base.twinq(states_v, acts_v)
        #
        # # states_v.shape: [128, 3]
        # # actions_v.shape: [128]
        # # target_action_values_v.shape: [128]
        # # torch.min(q1_v, q2_v).squeeze().shape: [128]
        # # self.calc_entropy(logstd_v=logstd_v).sum(dim=1).shape: [128]
        # target_values_v = torch.min(q1_v, q2_v).squeeze() - self.params.ENTROPY_LOSS_WEIGHT * self.calc_entropy(logstd_v=logstd_v).sum(dim=1)
        return states_v, actions_v, target_action_values_v
