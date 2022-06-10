import torch
from torch import nn
from d_agents.off_policy.tdmpc import helper as h


class TOLD(nn.Module):
    """Task-Oriented Latent Dynamics (TOLD) model used in TD-MPC."""
    def __init__(self, config, observation_shape, n_out_actions):
        super().__init__()
        self.config = config
        self._encoder = h.enc(config, observation_shape)
        self._dynamics = h.mlp(config.LATENT_DIM + n_out_actions, config.MLP_DIM, config.LATENT_DIM)
        self._reward = h.mlp(config.LATENT_DIM + n_out_actions, config.MLP_DIM, 1)
        self._pi = h.mlp(config.LATENT_DIM, config.MLP_DIM, n_out_actions)
        self._Q1, self._Q2 = h.q(config), h.q(config)
        # 모든 model들의 weight는 orthogonal matrix로 그리고 bias는 모두 0으로 초기화
        self.apply(h.orthogonal_init)
        # reward와 q model들의 마지막 layer는 weight와 bias를 모두 0으로 초기화
        for m in [self._reward, self._Q1, self._Q2]:
            m[-1].weight.data.fill_(0)
            m[-1].bias.data.fill_(0)

    def track_q_grad(self, enable=True):
        """Utility function. Enables/disables gradient tracking of Q-networks."""
        for m in [self._Q1, self._Q2]:
            h.set_requires_grad(m, enable)

    def h(self, obs):
        """Encodes an observation into its latent representation (h)."""
        return self._encoder(obs)

    def next(self, z, a):
        """Predicts next latent state (d) and single-step reward (R)."""
        x = torch.cat([z, a], dim=-1)
        return self._dynamics(x), self._reward(x)

    def pi(self, z, std=0):
        """Samples an action from the learned policy (pi)."""
        mu = torch.tanh(self._pi(z))
        if std > 0:
            std = torch.ones_like(mu) * std
            return h.TruncatedNormal(mu, std).sample(clip=0.3)
        return mu

    def Q(self, z, a):
        """Predict state-action value (Q)."""
        x = torch.cat([z, a], dim=-1)
        return self._Q1(x), self._Q2(x)
