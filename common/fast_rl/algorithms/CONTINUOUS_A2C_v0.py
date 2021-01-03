# -*- coding: utf-8 -*-
import math
import torch
import torch.nn.functional as F

from common.fast_rl import replay_buffer
from common.fast_rl.policy_based_model import unpack_batch_for_a2c
from config.names import DeepLearningModelName
from rl_main import rl_utils


class CONTINUOUS_A2C_FAST_v0:
    def __init__(self, env, worker_id, logger, params, device, verbose):
        self.env = env
        self.worker_id = worker_id
        self.params = params
        self.device = device
        self.logger = logger
        self.verbose = verbose

        assert self.params.DEEP_LEARNING_MODEL == DeepLearningModelName.CONTINUOUS_ACTOR_CRITIC_MLP
        self.model = rl_utils.get_rl_model(self.env, self.worker_id, params=self.params)

        print(self.model.base.actor)
        print(self.model.base.critic)

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

        if self.params.PER:
            self.buffer = replay_buffer.PrioReplayBuffer(
                experience_source=None, buffer_size=self.params.REPLAY_BUFFER_SIZE,
                n_step=self.params.N_STEP, beta_start=0.4, beta_frames=self.params.MAX_GLOBAL_STEP
            )
        else:
            self.buffer = replay_buffer.ExperienceReplayBuffer(
                experience_source=None, buffer_size=self.params.REPLAY_BUFFER_SIZE
            )

    def set_experience_source_to_buffer(self, experience_source):
        if self.params.PER:
            self.buffer = replay_buffer.PrioReplayBuffer(
                experience_source=experience_source, buffer_size=self.params.REPLAY_BUFFER_SIZE,
                n_step=self.params.N_STEP, beta_start=0.4, beta_frames=self.params.MAX_GLOBAL_STEP
            )
        else:
            self.buffer = replay_buffer.ExperienceReplayBuffer(
                experience_source=experience_source, buffer_size=self.params.REPLAY_BUFFER_SIZE
            )

    def train_net(self, step_idx):
        if self.params.PER:
            batch, batch_indices, batch_weights = self.buffer.sample(self.params.BATCH_SIZE)
        else:
            batch = self.buffer.sample(self.params.BATCH_SIZE)
            batch_indices, batch_weights = None, None

        # batch_states_v.shape: (32, 3)
        # batch_actions_v.shape: (32,)
        # batch_target_action_values_v.shape: (32,)
        batch_states_v, batch_actions_v, batch_target_action_values_v = unpack_batch_for_a2c(
            batch, self.model, self.params, device=self.device
        )

        batch.clear()

        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()

        # batch_mu_v.shape: (32, 1)
        # batch_var_v.shape: (32, 1)
        # batch_value_v.shape; (32, 1)
        batch_mu_v, batch_var_v, batch_value_v = self.model(batch_states_v)

        if self.params.PER:
            batch_loss_critic_v = F.mse_loss(batch_value_v.squeeze(-1), batch_target_action_values_v, reduction="none")
            batch_weights_v = torch.tensor(batch_weights)
            loss_critic_v = batch_weights_v * batch_loss_critic_v

            self.buffer.update_priorities(batch_indices, batch_loss_critic_v.detach().cpu().numpy() + 1e-5)
            self.buffer.update_beta(step_idx)
        else:
            loss_critic_v = F.mse_loss(batch_value_v.squeeze(-1), batch_target_action_values_v)

        loss_critic_v.backward(retain_graph=True)

        # batch_advantage_v.shape: (32, 1)
        batch_advantage_v = batch_target_action_values_v.unsqueeze(-1) - batch_value_v.detach()
        batch_log_prob_actions_v = batch_advantage_v * self.calc_log_prob(batch_mu_v, batch_var_v, batch_actions_v)
        loss_actor_v = -batch_log_prob_actions_v.mean()

        entropy_v = -1.0 * (torch.log(2 * math.pi * batch_var_v) + 1) / 2
        loss_entropy_v = self.params.ENTROPY_BETA * entropy_v.mean()

        # loss_actor_v를 작아지도록 만듦 --> batch_log_prob_actions_v.mean()가 커지도록 만듦
        # loss_entropy_v를 작아지도록 만듦 --> entropy_v가 커지도록 만듦
        #print(loss_critic_v, loss_actor_v, loss_entropy_v)
        loss_actor_and_entropy_v = loss_actor_v + loss_entropy_v
        loss_actor_and_entropy_v.backward()

        #nn_utils.clip_grad_norm_(self.model.parameters(), self.params.CLIP_GRAD)
        self.actor_optimizer.step()
        self.critic_optimizer.step()

        gradients = self.model.get_gradients_for_current_parameters()

        return gradients, loss_critic_v.item(), loss_actor_v.item() * -1.0

    @staticmethod
    def calc_log_prob(mu_v, var_v, actions_v):
        p1 = - ((mu_v - actions_v) ** 2) / (2 * var_v.clamp(min=1e-3))
        p2 = - torch.log(torch.sqrt(2 * math.pi * var_v))
        return p1 + p2