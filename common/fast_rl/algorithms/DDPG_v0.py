# -*- coding: utf-8 -*-
from collections import deque

import torch
import torch.nn.functional as F
import numpy as np

from common.fast_rl import rl_agent, replay_buffer
from common.fast_rl.policy_based_model import unpack_batch_for_ddpg
from rl_main import rl_utils


class DDPG_FAST_v0:
    def __init__(self, env, worker_id, logger, params, device, verbose):
        self.env = env
        self.worker_id = worker_id
        self.params = params
        self.device = device
        self.logger = logger
        self.verbose = verbose

        self.model = rl_utils.get_rl_model(self.env, self.worker_id, params=self.params)

        print(self.model.base.actor)
        print(self.model.base.critic)

        self.target_agent = rl_agent.TargetNet(self.model.base)

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

        # print(batch)
        batch_states_v, batch_actions_v, batch_rewards_v, batch_dones_mask, batch_last_states_v = unpack_batch_for_ddpg(
            batch, self.device
        )

        # train critic
        self.critic_optimizer.zero_grad()
        critic_parameters = self.model.base.critic.parameters()
        for p in critic_parameters:
            p.requires_grad = True

        batch_q_v = self.model.base.forward_critic(batch_states_v, batch_actions_v)
        batch_last_act_v = self.target_agent.target_model.forward_actor(batch_last_states_v)
        batch_q_last_v = self.target_agent.target_model.forward_critic(batch_last_states_v, batch_last_act_v)
        batch_q_last_v[batch_dones_mask] = 0.0
        batch_target_q_v = batch_rewards_v.unsqueeze(dim=-1) + batch_q_last_v * self.params.GAMMA ** self.params.N_STEP

        if self.params.PER:
            batch_l1_loss = F.smooth_l1_loss(batch_q_v, batch_target_q_v.detach(), reduction='none')  # for PER
            batch_weights_v = torch.tensor(batch_weights)
            critic_loss_v = batch_weights_v * batch_l1_loss

            self.buffer.update_priorities(batch_indices, batch_l1_loss.detach().cpu().numpy() + 1e-5)
            self.buffer.update_beta(step_idx)
        else:
            critic_loss_v = F.smooth_l1_loss(batch_q_v, batch_target_q_v.detach())

        loss_critic_v = critic_loss_v.mean()

        loss_critic_v.backward()
        self.critic_optimizer.step()

        # train actor
        self.actor_optimizer.zero_grad()
        critic_parameters = self.model.base.critic.parameters()
        for p in critic_parameters:
            p.requires_grad = False

        batch_current_actions_v = self.model.base.forward_actor(batch_states_v)
        actor_loss_v = -1.0 * self.model.base.forward_critic(batch_states_v, batch_current_actions_v)
        loss_actor_v = actor_loss_v.mean()

        loss_actor_v.backward()

        self.actor_optimizer.step()

        self.target_agent.alpha_sync(alpha=1 - 0.001)

        gradients = self.model.get_gradients_for_current_parameters()

        return gradients, loss_critic_v.item(), loss_actor_v.item() * -1.0
