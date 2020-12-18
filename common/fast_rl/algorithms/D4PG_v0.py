# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F
import numpy as np

from common.fast_rl import rl_agent, experience
from common.fast_rl.common.utils import distribution_projection
from common.fast_rl.policy_based_model import unpack_batch_for_ddpg
from rl_main import rl_utils


class D4PG_FAST_v0:
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
            self.buffer = experience.PrioReplayBuffer(
                experience_source=None, buffer_size=self.params.REPLAY_BUFFER_SIZE,
                n_step=self.params.N_STEP, beta_start=0.4, beta_frames=self.params.MAX_GLOBAL_STEPS
            )
        else:
            self.buffer = experience.ExperienceReplayBuffer(
                experience_source=None, buffer_size=self.params.REPLAY_BUFFER_SIZE
            )

    def set_experience_source_to_buffer(self, experience_source):
        if self.params.PER:
            self.buffer = experience.PrioReplayBuffer(
                experience_source=experience_source, buffer_size=self.params.REPLAY_BUFFER_SIZE,
                n_step=self.params.N_STEP, beta_start=0.4, beta_frames=self.params.MAX_GLOBAL_STEPS
            )
        else:
            self.buffer = experience.ExperienceReplayBuffer(
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

        batch_distribution_v = self.model.base.forward_critic(batch_states_v, batch_actions_v)
        batch_last_act_v = self.target_agent.target_model.forward_actor(batch_last_states_v)
        batch_last_distribution_v = F.softmax(
            self.target_agent.target_model.forward_critic(batch_last_states_v, batch_last_act_v), dim=1
        )

        proj_distr_v = distribution_projection(
            batch_last_distribution_v, batch_rewards_v, batch_dones_mask,
            self.params.V_MIN, self.params.V_MAX, self.params.N_ATOMS,
            gamma=self.params.GAMMA ** self.params.N_STEP, device=self.device
        )

        batch_q_prob_distribution_v = -F.log_softmax(batch_distribution_v, dim=1) * proj_distr_v
        loss_critic_v = batch_q_prob_distribution_v.sum(dim=1).mean()
        loss_critic_v.backward()
        self.critic_optimizer.step()

        # train actor
        self.actor_optimizer.zero_grad()
        critic_parameters = self.model.base.critic.parameters()
        for p in critic_parameters:
            p.requires_grad = False

        batch_current_actions_v = self.model.base.forward_actor(batch_states_v)
        batch_q_distribution_v = self.model.base.forward_critic(batch_states_v, batch_current_actions_v)
        actor_loss_v = -1.0 * self.model.base.distribution_to_q(batch_q_distribution_v)
        loss_actor_v = actor_loss_v.mean()

        loss_actor_v.backward()

        self.actor_optimizer.step()

        self.target_agent.alpha_sync(alpha=1 - 0.001)

        gradients = self.model.get_gradients_for_current_parameters()

        return gradients, loss_critic_v.item(), loss_actor_v.item() * -1.0