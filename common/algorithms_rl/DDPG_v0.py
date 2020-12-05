# -*- coding: utf-8 -*-
from collections import deque

import torch
from torch import optim
import torch.nn.functional as F
import numpy as np

from common.fast_rl import policy_based_model, rl_agent, experience, actions
from common.fast_rl.experience import ExperienceWithNoise
from common.fast_rl.policy_based_model import unpack_batch_for_ddpg
from common.fast_rl.rl_agent import float32_preprocessor
from config.names import DeepLearningModelName


class DDPG_v0:
    def __init__(self, env, worker_id, logger, params, device, verbose):
        self.env = env
        self.worker_id = worker_id
        self.params = params
        self.device = device
        self.logger = logger
        self.verbose = verbose

        if params.DEEP_LEARNING_MODEL is DeepLearningModelName.DDPG_MLP:
            self.actor_net = policy_based_model.DDPGActor(
                obs_size=3,
                hidden_size_1=512, hidden_size_2=256,
                n_actions=1,
                scale=2.0
            ).to(device)

            self.critic_net = policy_based_model.DDPGCritic(
                obs_size=3,
                hidden_size_1=512, hidden_size_2=256,
                n_actions=1
            ).to(device)
        elif params.DEEP_LEARNING_MODEL is DeepLearningModelName.DDPG_GRU:
            self.actor_net = policy_based_model.DDPGGruActor(
                obs_size=3,
                hidden_size_1=256, hidden_size_2=256,
                n_actions=1,
                bidirectional=False,
                scale=2.0
            ).to(device)

            self.critic_net = policy_based_model.DDPGGruCritic(
                obs_size=3,
                hidden_size_1=256, hidden_size_2=256,
                n_actions=1,
                bidirectional=False
            ).to(device)
        elif params.DEEP_LEARNING_MODEL is DeepLearningModelName.DDPG_GRU_ATTENTION:
            self.actor_net = policy_based_model.DDPGGruAttentionActor(
                obs_size=3,
                hidden_size=256,
                n_actions=1,
                bidirectional=False,
                scale=2.0
            ).to(device)

            self.critic_net = policy_based_model.DDPGGruAttentionCritic(
                obs_size=3,
                hidden_size_1=256, hidden_size_2=256,
                n_actions=1,
                bidirectional=False
            ).to(device)
        else:
            raise ValueError()

        print(self.actor_net)
        print(self.critic_net)

        self.target_actor_net = rl_agent.TargetNet(self.actor_net)
        self.target_critic_net = rl_agent.TargetNet(self.critic_net)

        self.actor_optimizer = optim.Adam(self.actor_net.parameters(), lr=params.ACTOR_LEARNING_RATE)
        self.critic_optimizer = optim.Adam(self.critic_net.parameters(), lr=params.LEARNING_RATE)

        if params.PER:
            self.buffer = experience.PrioReplayBuffer(
                experience_source=None, buffer_size=params.REPLAY_BUFFER_SIZE, n_step=params.N_STEP,
                beta_start=0.4, beta_frames=params.MAX_GLOBAL_STEPS
            )
        else:
            self.buffer = experience.ExperienceReplayBuffer(experience_source=None, buffer_size=params.REPLAY_BUFFER_SIZE)

    def train_net(self):
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
        batch_q_v = self.critic_net(batch_states_v, batch_actions_v)
        batch_last_act_v = self.target_actor_net.target_model(batch_last_states_v)
        batch_q_last_v = self.target_critic_net.target_model(batch_last_states_v, batch_last_act_v)
        batch_q_last_v[batch_dones_mask] = 0.0
        batch_target_q_v = batch_rewards_v.unsqueeze(dim=-1) + batch_q_last_v * self.params.GAMMA ** self.params.N_STEP

        if self.params.PER:
            batch_l1_loss = F.smooth_l1_loss(batch_q_v, batch_target_q_v.detach(), reduction='none')  # for PER
            batch_weights_v = torch.tensor(batch_weights)
            loss_critic_v = batch_weights_v * batch_l1_loss

            self.buffer.update_priorities(batch_indices, batch_l1_loss.detach().cpu().numpy() + 1e-5)
            self.buffer.update_beta(self.step_idx)
        else:
            loss_critic_v = F.smooth_l1_loss(batch_q_v, batch_target_q_v.detach())

        loss_critic_v.mean().backward()
        self.critic_optimizer.step()

        # train actor
        self.actor_optimizer.zero_grad()
        batch_current_actions_v = self.actor_net(batch_states_v)
        actor_loss_v = -self.critic_net(batch_states_v, batch_current_actions_v)
        loss_actor_v = actor_loss_v.mean()
        loss_actor_v.backward()

        self.actor_optimizer.step()

        self.target_actor_net.alpha_sync(alpha=1 - 0.001)
        self.target_critic_net.alpha_sync(alpha=1 - 0.001)