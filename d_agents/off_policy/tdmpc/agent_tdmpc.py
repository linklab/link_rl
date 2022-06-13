# https://github.com/pranz24/pytorch-soft-actor-critic/blob/master/sac.py
# https://github.com/BY571/Soft-Actor-Critic-and-Extensions/blob/master/SAC.py
# PAPER: https://arxiv.org/abs/1812.05905
# https://www.pair.toronto.edu/csc2621-w20/assets/slides/lec4_sac.pdf
# https://bair.berkeley.edu/blog/2017/10/06/soft-q-learning/
from copy import deepcopy

import torch.optim as optim
import numpy as np
import torch
import torch.multiprocessing as mp
from torch.distributions import Normal

from c_models.h_sac_models import ContinuousSacModel
from c_models.j_tdmpc_models import TOLD
from d_agents.off_policy.off_policy_agent import OffPolicyAgent
from g_utils.types import AgentMode
from d_agents.off_policy.tdmpc import helper as h
from d_agents.off_policy.tdmpc.tdmpc import TDMPC
import re
from d_agents.off_policy.tdmpc import helper


class AgentTdmpc(OffPolicyAgent):
    def __init__(self, observation_space, action_space, config):
        super(AgentTdmpc, self).__init__(observation_space, action_space, config)
        self.config = config
        self.device = torch.device('cuda')
        self.std = h.linear_schedule(config.STD_SCHEDULE, 0)
        self.model = TOLD(
            observation_shape=self.observation_shape, n_out_actions=self.n_out_actions, config=config
        ).to(config.DEVICE)
        self.model_target = deepcopy(self.model).to(config.DEVICE)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.config.LEARNING_RATE)
        self.pi_optim = torch.optim.Adam(self.model._pi.parameters(), lr=self.config.LEARNING_RATE)
        self.aug = h.RandomShiftsAug(config)
        self.model.eval()
        self.model_target.eval()

    def get_action(self, obs, mode=AgentMode.TRAIN, t0=False, step=None):
        action = self.plan(obs, mode, step, t0)
        return action

    def state_dict(self):
        """Retrieve state dict of TOLD model, including slow-moving target network."""
        return {'model': self.model.state_dict(),
                'model_target': self.model_target.state_dict()}

    def save(self, fp):
        """Save state dict of TOLD model to filepath."""
        torch.save(self.state_dict(), fp)

    def load(self, fp):
        """Load a saved state dict from filepath into current agent."""
        d = torch.load(fp)
        self.model.load_state_dict(d['model'])
        self.model_target.load_state_dict(d['model_target'])

    @torch.no_grad()
    def estimate_value(self, z, actions, horizon):
        """Estimate value of a trajectory starting at latent state z and executing given actions."""
        G, discount = 0, 1
        for t in range(horizon):
            z, reward = self.model.next(z, actions[t])
            G += discount * reward
            discount *= self.config.GAMMA
        G += discount * torch.min(*self.model.Q(z, self.model.pi(z, self.config.MIN_STD)))
        return G

    @torch.no_grad()
    def plan(self, obs, mode=AgentMode.TRAIN, step=None, t0=True):
        """
        Plan next action using TD-MPC inference.
        obs: raw input observation.
        eval_mode: uniform sampling and action noise is disabled during evaluation.
        step: current time step. determines e.g. planning horizon.
        t0: whether current step is the first step of an episode.
        """
        # Seed steps
        if step < self.config.TRAIN_INTERVAL_GLOBAL_TIME_STEPS and mode == AgentMode.TRAIN:
            return torch.empty(self.n_out_actions, dtype=torch.float32, device=self.device).uniform_(-1, 1)

        # Sample policy trajectories
        obs = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        horizon = int(min(self.config.HORIZON, h.linear_schedule(self.config.HORIZON_SCHEDULE, step)))
        num_pi_trajs = int(self.config.MIXTURE_COEF * self.config.NUM_SAMPLES)
        if num_pi_trajs > 0:
            pi_actions = torch.empty(horizon, num_pi_trajs, self.n_out_actions, device=self.device)
            z = self.model.h(obs).repeat(num_pi_trajs, 1)
            for t in range(horizon):
                pi_actions[t] = self.model.pi(z, self.config.MIN_STD)
                z, _ = self.model.next(z, pi_actions[t])

        # Initialize state and parameters
        z = self.model.h(obs).repeat(self.config.NUM_SAMPLES + num_pi_trajs, 1)
        mean = torch.zeros(horizon, self.n_out_actions, device=self.device)
        std = 2 * torch.ones(horizon, self.n_out_actions, device=self.device)
        if not t0 and hasattr(self, '_prev_mean') and mode == AgentMode.TRAIN:
            mean[:-1] = self._train_prev_mean[1:]
        elif not t0 and hasattr(self, '_prev_mean') and mode == AgentMode.TEST:
            mean[:-1] = self._test_prev_mean[1:]

        # Iterate CEM
        for i in range(self.config.ITERATION):
            actions = torch.clamp(mean.unsqueeze(1) + std.unsqueeze(1) * \
                                  torch.randn(horizon, self.config.NUM_SAMPLES, self.n_out_actions, device=std.device),
                                  -1, 1)
            if num_pi_trajs > 0:
                actions = torch.cat([actions, pi_actions], dim=1)

            # Compute elite actions
            value = self.estimate_value(z, actions, horizon)
            elite_idxs = torch.topk(value.squeeze(1), self.config.NUM_ELITES, dim=0).indices
            elite_value, elite_actions = value[elite_idxs], actions[:, elite_idxs]

            # Update parameters
            max_value = elite_value.max(0)[0]
            score = torch.exp(self.config.TEMPERATURE * (elite_value - max_value))
            score /= score.sum(0)
            _mean = torch.sum(score.unsqueeze(0) * elite_actions, dim=1) / (score.sum(0) + 1e-9)
            _std = torch.sqrt(torch.sum(score.unsqueeze(0) * (elite_actions - _mean.unsqueeze(1)) ** 2, dim=1) / (
                    score.sum(0) + 1e-9))
            _std = _std.clamp_(self.std, 2)
            mean, std = self.config.MOMENTUM * mean + (1 - self.config.MOMENTUM) * _mean, _std

        # Outputs
        score = score.squeeze(1).cpu().numpy()
        actions = elite_actions[:, np.random.choice(np.arange(score.shape[0]), p=score)]
        mean, std = actions[0], _std[0]
        a = mean
        if mode == AgentMode.TRAIN:
            a += std * torch.randn(self.n_out_actions, device=std.device)
            self._train_prev_mean = mean
        else:
            self._test_prev_mean = mean
        return a

    def update_pi(self, zs):
        """Update policy using a sequence of latent states."""
        self.pi_optim.zero_grad(set_to_none=True)
        self.model.track_q_grad(False)

        # Loss is a weighted sum of Q-values
        pi_loss = 0
        for t, z in enumerate(zs):
            a = self.model.pi(z, self.config.MIN_STD)
            Q = torch.min(*self.model.Q(z, a))
            pi_loss += -Q.mean() * (self.config.RHO ** t)

        pi_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model._pi.parameters(), self.config.CLIP_GRADIENT_VALUE, error_if_nonfinite=False)
        self.pi_optim.step()
        self.model.track_q_grad(True)
        return pi_loss.item()

    @torch.no_grad()
    def _td_target(self, next_obs, reward):
        """Compute the TD-target from a reward and the observation at the following time step."""
        next_z = self.model.h(next_obs)
        td_target = reward + self.config.GAMMA * \
                    torch.min(*self.model_target.Q(next_z, self.model.pi(next_z, self.config.MIN_STD)))
        return td_target

    def train_tdmpc(self, step):
        """Main update function. Corresponds to one iteration of the TOLD model learning."""
        obs, next_obses, action, reward, idxs, weights = \
            self.observations, self.next_observations, self.actions, self.rewards, self.idx, self.weifhts
        self.optim.zero_grad(set_to_none=True)
        self.std = h.linear_schedule(self.config.STD_SCHEDULE, step)
        self.model.train()

        # Representation
        z = self.model.h(self.aug(obs))
        zs = [z.detach()]

        consistency_loss, reward_loss, value_loss, priority_loss = 0, 0, 0, 0
        for t in range(self.config.HORIZON):
            # Predictions
            Q1, Q2 = self.model.Q(z, action[t])
            z, reward_pred = self.model.next(z, action[t])
            with torch.no_grad():
                next_obs = self.aug(next_obses[t])
                next_z = self.model_target.h(next_obs)
                td_target = self._td_target(next_obs, reward[t])
            zs.append(z.detach())

            # Losses
            rho = (self.config.RHO ** t)
            consistency_loss += rho * torch.mean(h.mse(z, next_z), dim=1, keepdim=True)
            reward_loss += rho * h.mse(reward_pred, reward[t])
            value_loss += rho * (h.mse(Q1, td_target) + h.mse(Q2, td_target))
            priority_loss += rho * (h.l1(Q1, td_target) + h.l1(Q2, td_target))

        # Optimize model
        total_loss = self.config.COSISTENCY_COEF * consistency_loss.clamp(max=1e4) + \
                     self.config.REWARD_COEF * reward_loss.clamp(max=1e4) + \
                     self.config.VALUE_COEF * value_loss.clamp(max=1e4)
        weighted_loss = (total_loss * weights).mean()
        weighted_loss.register_hook(lambda grad: grad * (1 / self.config.HORIZON))
        weighted_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.CLIP_GRADIENT_VALUE,
                                                   error_if_nonfinite=False)
        self.optim.step()
        self.replay_buffer.update_priorities(idxs, priority_loss.clamp(max=1e4).detach())

        # Update policy + target network
        pi_loss = self.update_pi(zs)
        if step % self.config.UPDATE_FREQ == 0:
            h.ema(self.model, self.model_target, self.config.TAU)

        self.model.eval()
        return {'consistency_loss': float(consistency_loss.mean().item()),
                'reward_loss': float(reward_loss.mean().item()),
                'value_loss': float(value_loss.mean().item()),
                'pi_loss': pi_loss,
                'total_loss': float(total_loss.mean().item()),
                'weighted_loss': float(weighted_loss.mean().item()),
                'grad_norm': float(grad_norm)}
