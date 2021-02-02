import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Normal
import torch.nn.utils as nn_utils

from codes.d_agents.a0_base_agent import BaseAgent, float32_preprocessor, TargetNet
from codes.d_agents.on_policy.on_policy_agent import OnPolicyAgent
from codes.e_utils import rl_utils, replay_buffer
from codes.e_utils.actions import ContinuousNormalActionSelector
from codes.e_utils.names import DeepLearningModelName, AgentMode


class AgentSAC(OnPolicyAgent):
    """
    """
    def __init__(
            self, worker_id, input_shape, num_outputs, action_min, action_max, params, device
    ):
        assert params.DEEP_LEARNING_MODEL == DeepLearningModelName.SOFT_ACTOR_CRITIC_MLP

        super(AgentSAC, self).__init__(params, device)
        self.__name__ = "AgentSAC"
        self.worker_id = worker_id
        self.action_min = action_min
        self.action_max = action_max

        self.train_action_selector = ContinuousNormalActionSelector()
        self.test_and_play_action_selector = ContinuousNormalActionSelector()

        self.model = rl_utils.get_rl_model(
            worker_id=worker_id, input_shape=input_shape, num_outputs=num_outputs, params=params, device=self.device
        )

        self.target_agent = TargetNet(self.model.base)

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

        self.twinq_optimizer = rl_utils.get_optimizer(
            parameters=self.model.base.twinq.parameters(),
            learning_rate=self.params.LEARNING_RATE,
            params=params
        )

        self.buffer = replay_buffer.ExperienceReplayBuffer(experience_source=None, buffer_size=self.params.BATCH_SIZE)

    def __call__(self, states, agent_states=None):
        if not isinstance(states, torch.FloatTensor):
            states = float32_preprocessor(states).to(self.device)

        if len(states) == 1:
            self.model.eval()
        else:
            self.model.train()

        mu_v, values_v = self.model(states)

        if self.agent_mode == AgentMode.TRAIN:
            actions = self.train_action_selector(mu_v, self.model.base.actor.logstd, self.action_min, self.action_max)
        else:
            actions = self.test_and_play_action_selector(
                mu_v, self.model.base.actor.logstd, self.action_min, self.action_max
            )

        critics = values_v.data.cpu().numpy()

        return actions, critics

    def train(self, step_idx):
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
        q1_loss_v = F.mse_loss(q1_v.squeeze(), target_action_values_v.detach())
        q2_loss_v = F.mse_loss(q2_v.squeeze(), target_action_values_v.detach())
        q_loss_v = q1_loss_v + q2_loss_v
        #q_loss_v = q_loss_v.mean()
        q_loss_v.backward()
        nn_utils.clip_grad_norm_(self.model.base.twinq.parameters(), self.params.CLIP_GRAD)
        self.twinq_optimizer.step()

        # train critic
        self.critic_optimizer.zero_grad()
        val_v = self.model.base.critic(states_v)

        if self.params.PER:
            batch_l1_loss = F.smooth_l1_loss(val_v.squeeze(), target_values_v.detach(), reduction="none")
            batch_weights_v = torch.tensor(batch_weights)
            critic_loss_v = batch_weights_v * batch_l1_loss

            self.buffer.update_priorities(batch_indices, batch_l1_loss.detach().cpu().numpy() + 1e-5)
            self.buffer.update_beta(step_idx)
        else:
            critic_loss_v = F.smooth_l1_loss(val_v.squeeze(), target_values_v.detach(), reduction="none")

        loss_critic_v = critic_loss_v.mean()
        loss_critic_v.backward()
        nn_utils.clip_grad_norm_(self.model.base.critic.parameters(), self.params.CLIP_GRAD)
        self.critic_optimizer.step()

        # train actor
        self.actor_optimizer.zero_grad()
        current_actions_v = self.model.base.actor(states_v)
        q1_v, q2_v = self.model.base.twinq(states_v, current_actions_v)
        loss_actor_v = -1.0 * torch.min(q1_v, q2_v).squeeze().mean()
        loss_actor_v.backward()
        nn_utils.clip_grad_norm_(self.model.base.actor.parameters(), self.params.CLIP_GRAD)
        self.actor_optimizer.step()

        self.target_agent.alpha_sync(alpha=1 - 0.001)

        gradients = self.model.get_gradients_for_current_parameters()

        return gradients, loss_critic_v.item(), loss_actor_v.item() * -1.0

    def unpack_batch_for_sac(self, batch):
        states_v, actions_v, target_action_values_v = self.unpack_batch_for_actor_critic(
            batch, self.model, self.params, device=self.device
        )

        # references for the critic network
        mu_v = self.model.base.actor(states_v)
        act_dist = Normal(mu_v, torch.exp(self.model.base.actor.logstd))
        acts_v = act_dist.sample()
        q1_v, q2_v = self.model.base.twinq(states_v, acts_v)
        # element-wise minimum
        target_values_v = torch.min(q1_v, q2_v).squeeze() - self.params.ENTROPY_LOSS_WEIGHT * act_dist.log_prob(acts_v).sum(dim=1)
        return states_v, actions_v, target_values_v, target_action_values_v