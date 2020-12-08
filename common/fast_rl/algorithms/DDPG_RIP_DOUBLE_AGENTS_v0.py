# -*- coding: utf-8 -*-
from collections import deque

import torch
from torch import optim
import torch.nn.functional as F

from common.fast_rl import rl_agent, experience
from common.fast_rl.policy_based_model import unpack_batch_for_ddpg
from rl_main import rl_utils


class DDPG_RIP_DOUBLE_AGENTS_v0:
    def __init__(self, env, worker_id, logger, params, device, verbose):
        self.env = env
        self.worker_id = worker_id
        self.params = params
        self.device = device
        self.logger = logger
        self.verbose = verbose

        ###########################
        ### SWING_UP Controller ###
        ###########################
        self.swing_up_model = rl_utils.get_rl_model(self.env, self.worker_id, params=self.params)

        print(self.swing_up_model.base.actor)
        print(self.swing_up_model.base.critic)

        self.swing_up_target_agent = rl_agent.TargetNet(self.swing_up_model.base)

        self.swing_up_actor_optimizer = rl_utils.get_optimizer(
            parameters=self.swing_up_model.base.actor.parameters(),
            learning_rate=self.params.ACTOR_LEARNING_RATE,
            params=params
        )

        self.swing_up_critic_optimizer = rl_utils.get_optimizer(
            parameters=self.swing_up_model.base.critic.parameters(),
            learning_rate=self.params.LEARNING_RATE,
            params=params
        )

        ############################
        ### BALANCING Controller ###
        ############################
        self.balancing_model = rl_utils.get_rl_model(self.env, self.worker_id, params=self.params)

        print(self.balancing_model.base.actor)
        print(self.balancing_model.base.critic)

        self.balancing_target_agent = rl_agent.TargetNet(self.balancing_model.base)

        self.balancing_actor_optimizer = rl_utils.get_optimizer(
            parameters=self.balancing_model.base.actor.parameters(),
            learning_rate=self.params.ACTOR_LEARNING_RATE,
            params=params
        )

        self.balancing_critic_optimizer = rl_utils.get_optimizer(
            parameters=self.balancing_model.base.critic.parameters(),
            learning_rate=self.params.LEARNING_RATE,
            params=params
        )

    def set_buffer(self, experience_source):
        if self.params.PER:
            self.swing_up_buffer = experience.PrioritizedReplayBuffer(
                experience_source=experience_source, buffer_size=self.params.REPLAY_BUFFER_SIZE,
                n_step=self.params.N_STEP, beta_start=0.4, beta_frames=self.params.MAX_GLOBAL_STEPS
            )
            self.balancing_buffer = experience.PrioritizedReplayBuffer(
                experience_source=experience_source, buffer_size=self.params.REPLAY_BUFFER_SIZE,
                n_step=self.params.N_STEP, beta_start=0.4, beta_frames=self.params.MAX_GLOBAL_STEPS
            )
        else:
            self.swing_up_buffer = experience.ExperienceReplayBuffer(
                experience_source=experience_source, buffer_size=self.params.REPLAY_BUFFER_SIZE
            )
            self.balancing_buffer = experience.ExperienceReplayBuffer(
                experience_source=experience_source, buffer_size=self.params.REPLAY_BUFFER_SIZE
            )

    def train_swing_up_net(self, step_idx):
        gradients, loss = self._train_net(
            self.swing_up_buffer, self.swing_up_model,
            self.swing_up_actor_optimizer, self.swing_up_critic_optimizer, self.swing_up_target_agent, step_idx
        )
        return gradients, loss

    def train_balancing_net(self, step_idx):
        gradients, loss = self._train_net(
            self.balancing_buffer, self.balancing_model,
            self.balancing_actor_optimizer, self.balancing_critic_optimizer, self.balancing_target_agent, step_idx
        )
        return gradients, loss

    def _train_net(self, buffer, model, actor_optimizer, critic_optimizer, target_agent, step_idx):
        if self.params.PER:
            batch, batch_indices, batch_weights = buffer.sample(self.params.BATCH_SIZE)
        else:
            batch = buffer.sample(self.params.BATCH_SIZE)
            batch_indices, batch_weights = None, None

        # print(batch)
        batch_states_v, batch_actions_v, batch_rewards_v, batch_dones_mask, batch_last_states_v = unpack_batch_for_ddpg(
            batch, self.device
        )

        # train critic
        critic_optimizer.zero_grad()
        batch_q_v = model.base.forward_critic(batch_states_v, batch_actions_v)
        batch_last_act_v = target_agent.target_model.forward_actor(batch_last_states_v)
        batch_q_last_v = target_agent.target_model.forward_critic(batch_last_states_v, batch_last_act_v)
        batch_q_last_v[batch_dones_mask] = 0.0
        batch_target_q_v = batch_rewards_v.unsqueeze(dim=-1) + batch_q_last_v * self.params.GAMMA ** self.params.N_STEP

        if self.params.PER:
            batch_l1_loss = F.smooth_l1_loss(batch_q_v, batch_target_q_v.detach(), reduction='none')  # for PER
            batch_weights_v = torch.tensor(batch_weights)
            critic_loss_v = batch_weights_v * batch_l1_loss

            buffer.update_priorities(batch_indices, batch_l1_loss.detach().cpu().numpy() + 1e-5)
            buffer.update_beta(step_idx)
        else:
            critic_loss_v = F.smooth_l1_loss(batch_q_v, batch_target_q_v.detach())

        loss_critic_v = critic_loss_v.mean()
        loss_critic_v.backward()
        critic_optimizer.step()

        # train actor
        actor_optimizer.zero_grad()
        batch_current_actions_v = model.base.forward_actor(batch_states_v)
        actor_loss_v = -model.base.forward_critic(batch_states_v, batch_current_actions_v)
        loss_actor_v = actor_loss_v.mean()
        loss_actor_v.backward()

        actor_optimizer.step()

        target_agent.alpha_sync(alpha=1 - 0.001)

        gradients = model.get_gradients_for_current_parameters()
        loss = loss_critic_v + loss_actor_v

        return gradients, loss.item()