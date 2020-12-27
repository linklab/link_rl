# -*- coding: utf-8 -*-
from collections import deque

import torch
from torch import optim
import torch.nn.functional as F

from common.fast_rl import rl_agent, experience, value_based_model, replay_buffer
from common.fast_rl.policy_based_model import unpack_batch_for_ddpg
from rl_main import rl_utils


class Dueling_Double_DQN_v0:
    def __init__(self, env, worker_id, logger, params, device, verbose):
        self.env = env
        self.worker_id = worker_id
        self.params = params
        self.device = device
        self.logger = logger
        self.verbose = verbose

        self.model = rl_utils.get_rl_model(self.env, self.worker_id, params=self.params)

        print(self.model)

        self.target_agent = rl_agent.TargetNet(self.model.base)

        self.optimizer = rl_utils.get_optimizer(
            parameters=self.model.base.parameters(),
            learning_rate=self.params.ACTOR_LEARNING_RATE,
            params=params
        )

        if self.params.PER:
            self.buffer = replay_buffer.PrioReplayBuffer(
                experience_source=None, buffer_size=self.params.REPLAY_BUFFER_SIZE,
                n_step=self.params.N_STEP, beta_start=0.4, beta_frames=self.params.MAX_GLOBAL_STEPS
            )
        else:
            self.buffer = replay_buffer.ExperienceReplayBuffer(
                experience_source=None, buffer_size=self.params.REPLAY_BUFFER_SIZE
            )

    def set_experience_source_to_buffer(self, experience_source):
        if self.params.PER:
            self.buffer = replay_buffer.PrioReplayBuffer(
                experience_source=experience_source, buffer_size=self.params.REPLAY_BUFFER_SIZE,
                n_step=self.params.N_STEP, beta_start=0.4, beta_frames=self.params.MAX_GLOBAL_STEPS
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

        self.optimizer.zero_grad()


        if self.params.PER:
            loss_v, sample_prios = value_based_model.calc_loss_per_double_dqn(
                self.buffer.buffer, batch, batch_indices, batch_weights,
                self.model, self.target_agent, self.params, cuda=self.params.CUDA, cuda_async=True
            )
            self.buffer.update_priorities(batch_indices, sample_prios.data.cpu().numpy())
            self.buffer.update_beta(step_idx)
        else:
            loss_v = value_based_model.calc_loss_double_dqn(
                batch, self.model, self.target_agent, self.params.GAMMA, cuda=self.params.CUDA, cuda_async=False
            )

        loss_v.backward()
        self.optimizer.step()

        if step_idx % self.params.TARGET_NET_SYNC_STEP_PERIOD < self.params.TRAIN_STEP_FREQ:
            self.target_agent.sync()

        gradients = self.model.get_gradients_for_current_parameters()

        return gradients, loss_v.item()