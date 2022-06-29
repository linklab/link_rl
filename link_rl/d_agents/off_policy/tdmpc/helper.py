import re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions as pyd
from torch.distributions.utils import _standard_normal

__REDUCE__ = lambda b: 'mean' if b else 'none'

from link_rl.g_utils.commons_rl import Episode


def l1(pred, target, reduce=False):
    """Computes the L1-loss between predictions and targets."""
    return F.l1_loss(pred, target, reduction=__REDUCE__(reduce))


def mse(pred, target, reduce=False):
    """Computes the MSE loss between predictions and targets."""
    return F.mse_loss(pred, target, reduction=__REDUCE__(reduce))


def _get_out_shape(in_shape, layers):
    """Utility function. Returns the output shape of a network for a given input shape."""
    x = torch.randn(*in_shape).unsqueeze(0)
    return (nn.Sequential(*layers) if isinstance(layers, list) else layers)(x).squeeze(0).shape


def orthogonal_init(m):
    """Orthogonal layer initialization."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data, gain)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def ema(m, m_target, tau):
    """Update slow-moving average of online network (target network) at rate tau."""
    with torch.no_grad():
        for p, p_target in zip(m.parameters(), m_target.parameters()):
            p_target.data.lerp_(p.data, tau)


def set_requires_grad(net, value):
    """Enable/disable gradients for a given (sub)network."""
    for param in net.parameters():
        param.requires_grad_(value)


class TruncatedNormal(pyd.Normal):
    """Utility class implementing the truncated normal distribution."""

    def __init__(self, loc, scale, low=-1.0, high=1.0, eps=1e-6):
        super().__init__(loc, scale, validate_args=False)
        self.low = low
        self.high = high
        self.eps = eps

    def _clamp(self, x):
        clamped_x = torch.clamp(x, self.low + self.eps, self.high - self.eps)
        x = x - x.detach() + clamped_x.detach()
        return x

    def sample(self, clip=None, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        eps = _standard_normal(shape,
                               dtype=self.loc.dtype,
                               device=self.loc.device)
        eps *= self.scale
        if clip is not None:
            eps = torch.clamp(eps, -clip, clip)
        x = self.loc + eps
        return self._clamp(x)


class NormalizeImg(nn.Module):
    """Normalizes pixel observations to [0,1) range."""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.div(255.)


class Flatten(nn.Module):
    """Flattens its input to a (batched) vector."""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


def enc(config, observation_shape):
    """Returns a TOLD encoder."""
    if config.FROM_PIXELS:
        C = int(3 * config.FRAME_STACK)
        layers = [nn.Conv2d(C, config.NUM_CHANNELS, 7, stride=2), nn.ReLU(),
                  nn.Conv2d(config.NUM_CHANNELS, config.NUM_CHANNELS, 5, stride=2), nn.ReLU(),
                  nn.Conv2d(config.NUM_CHANNELS, config.NUM_CHANNELS, 3, stride=2), nn.ReLU(),
                  nn.Conv2d(config.NUM_CHANNELS, config.NUM_CHANNELS, 3, stride=2), nn.ReLU()]
        out_shape = _get_out_shape((C, config.IMG_SIZE, config.IMG_SIZE), layers)
        layers.extend([Flatten(), nn.Linear(np.prod(out_shape), config.LATENT_DIM)])
    else:
        layers = [nn.Linear(*observation_shape, config.ENC_DIM), nn.ELU(),
                  nn.Linear(config.ENC_DIM, config.LATENT_DIM)]
    return nn.Sequential(*layers)


def mlp(in_dim, mlp_dim, out_dim, act_fn=nn.ELU()):
    """Returns an MLP."""
    if isinstance(mlp_dim, int):
        mlp_dim = [mlp_dim, mlp_dim]
    return nn.Sequential(
        nn.Linear(in_dim, mlp_dim[0]), act_fn,
        nn.Linear(mlp_dim[0], mlp_dim[1]), act_fn,
        nn.Linear(mlp_dim[1], out_dim))


def q(config, n_out_actions, act_fn=nn.ELU()):
    """Returns a Q-function that uses Layer Normalization."""
    return nn.Sequential(nn.Linear(config.LATENT_DIM + n_out_actions, config.MLP_DIM), nn.LayerNorm(config.MLP_DIM), nn.Tanh(),
                         nn.Linear(config.MLP_DIM, config.MLP_DIM), nn.ELU(),
                         nn.Linear(config.MLP_DIM, 1))


class RandomShiftsAug(nn.Module):
    """
    Random shift image augmentation.
    Adapted from https://github.com/facebookresearch/drqv2
    """

    def __init__(self, config):
        super().__init__()
        self.pad = int(config.IMG_SIZE / 21) if config.FROM_PIXELS else None

    def forward(self, x):
        if not self.pad:
            return x
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, 'replicate')
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps, 1.0 - eps, h + 2 * self.pad, device=x.device, dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)
        shift = torch.randint(0, 2 * self.pad + 1, size=(n, 1, 1, 2), device=x.device, dtype=x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)
        grid = base_grid + shift
        return F.grid_sample(x, grid, padding_mode='zeros', align_corners=False)





class ReplayBuffer():
    """
    Storage and sampling functionality for training TD-MPC / TOLD.
    The replay buffer is stored in GPU memory when training from state.
    Uses prioritized experience replay by default."""

    def __init__(self, config, observation_space, action_space):
        self.config = config
        self.capacity = min(config.MAX_TRAINING_STEPS, config.BUFFER_CAPACITY)
        dtype = torch.float32 if not config.FROM_PIXELS else torch.uint8
        obs_shape = observation_space.shape if not config.FROM_PIXELS else (3, *observation_space.shape[-2:])
        action_space = action_space.shape[0]
        self.episode_length = int(1000/config.ACTION_REPEAT)
        last_obs_first_shape = int(self.capacity // self.episode_length)
        self._obs = torch.empty((self.capacity + 1, *obs_shape), dtype=dtype, device=self.config.DEVICE)
        # _last_obs 에는 한 'episode_length+1' obs가 저장된다.
        self._last_obs = torch.empty((last_obs_first_shape, *observation_space.shape), dtype=dtype, device=self.config.DEVICE)
        self._action = torch.empty((self.capacity, action_space), dtype=torch.float32, device=self.config.DEVICE)
        self._reward = torch.empty((self.capacity,), dtype=torch.float32, device=self.config.DEVICE)
        self._priorities = torch.ones((self.capacity,), dtype=torch.float32, device=self.config.DEVICE)
        self._eps = 1e-6
        self._full = False
        self.idx = 0
        self.buffer_len = 0

        self.observation_shape = observation_space.shape
        self.n_out_action = action_space

    def __add__(self, episode: Episode):
        self.add(episode)
        return self

    def __len__(self):
        return self.buffer_len

    def append(self, episode: Episode):
        self.add(episode)
        return self

    def add(self, episode: Episode):
        self._obs[self.idx:self.idx + self.episode_length] = episode.obs[
                                                                 :-1] if not self.config.FROM_PIXELS else episode.obs[
                                                                                                           :-1, -3:]
        self._last_obs[self.idx // self.episode_length] = episode.obs[-1]
        self._action[self.idx:self.idx + self.episode_length] = episode.action
        self._reward[self.idx:self.idx + self.episode_length] = episode.reward
        if self._full:
            max_priority = self._priorities.max().to(self.config.DEVICE).item()
        else:
            max_priority = 1. if self.idx == 0 else self._priorities[:self.idx].max().to(self.config.DEVICE).item()
        mask = torch.arange(self.episode_length) >= self.episode_length - self.config.HORIZON
        new_priorities = torch.full((self.episode_length,), max_priority, device=self.config.DEVICE)
        new_priorities[mask] = 0
        self._priorities[self.idx:self.idx + self.episode_length] = new_priorities
        self.idx = (self.idx + self.episode_length) % self.capacity
        self._full = self._full or self.idx == 0
        self.buffer_len = min(self.buffer_len+self.episode_length, self.config.BUFFER_CAPACITY)

    def update_priorities(self, idxs, priorities):
        self._priorities[idxs] = priorities.squeeze(1).to(self.config.DEVICE) + self._eps

    def _get_obs(self, arr, idxs):
        if not self.config.FROM_PIXELS:
            return arr[idxs]
        obs = torch.empty((self.config.BATCH_SIZE, 3 * self.config.FRAME_STACK, *arr.shape[-2:]), dtype=arr.dtype,
                          device=self.config.DEVICE)
        obs[:, -3:] = arr[idxs].to(self.config.DEVICE)
        _idxs = idxs.clone()
        mask = torch.ones_like(_idxs, dtype=torch.bool)
        
        # done = True 일경우에는 자신의 obs를 frame stack 시킴
        for i in range(1, self.config.FRAME_STACK):
            mask[_idxs % self.episode_length == 0] = False  # episode 첫 idx의 mask를 False로 바꾼다.
            _idxs[mask] -= 1  # episode의 첫 idx를 제외한 모든 idx를 -1씩 한다.
            obs[:, -(i + 1) * 3:-i * 3] = arr[_idxs].to(self.config.DEVICE)
        return obs.float()

    def sample(self):
        probs = (self._priorities if self._full else self._priorities[:self.idx]) ** self.config.PER_ALPHA
        probs /= probs.sum()
        total = len(probs)
        idxs = torch.from_numpy(
            np.random.choice(total, self.config.BATCH_SIZE, p=probs.cpu().numpy(), replace=not self._full)).to(self.config.DEVICE)
        weights = (total * probs[idxs]) ** (-self.config.PER_BETA)
        weights /= weights.max()

        obs = self._get_obs(self._obs, idxs)
        next_obs_shape = self._last_obs.shape[1:] if not self.config.FROM_PIXELS else (3 * self.config.FRAME_STACK, *self._last_obs.shape[-2:])
        next_obs = torch.empty((self.config.HORIZON + 1, self.config.BATCH_SIZE, *next_obs_shape), dtype=obs.dtype,
                               device=self.config.DEVICE)
        action = torch.empty((self.config.HORIZON + 1, self.config.BATCH_SIZE, *self._action.shape[1:]), dtype=torch.float32,
                             device=self.config.DEVICE)
        reward = torch.empty((self.config.HORIZON + 1, self.config.BATCH_SIZE), dtype=torch.float32, device=self.config.DEVICE)
        for t in range(self.config.HORIZON + 1):
            _idxs = idxs + t
            next_obs[t] = self._get_obs(self._obs, _idxs + 1)
            action[t] = self._action[_idxs]
            reward[t] = self._reward[_idxs]

        mask = (_idxs + 1) % self.episode_length == 0  # episode 첫 스텝 mask
        # next_step이 episode의 첫 스텝 일 때(에피소드가 현재 obs에서 종료 되었을 때), self._last_obs를 준다.
        # self._last_obs = episode.obs[episode_length + 1], 즉 에피소드 종료 후에 한 스텝 더 간 obs
        next_obs[-1, mask] = self._last_obs[_idxs[mask] // self.episode_length].to(self.config.DEVICE).float()
        if not action.device == self.config.DEVICE:
            action, reward, idxs, weights = \
                action.to(self.config.DEVICE), reward.to(self.config.DEVICE), idxs.to(self.config.DEVICE), weights.to(self.config.DEVICE)

        return obs, next_obs, action, reward.unsqueeze(2), idxs, weights


def linear_schedule(schdl, step):
    """
    Outputs values following a linear decay schedule.
    Adapted from https://github.com/facebookresearch/drqv2
    """
    try:
        return float(schdl)
    except ValueError:
        match = re.match(r'linear\((.+),(.+),(.+)\)', schdl)
        if match:
            init, final, duration = [float(g) for g in match.groups()]
            mix = np.clip(step / duration, 0.0, 1.0)
            return (1.0 - mix) * init + mix * final
    raise NotImplementedError(schdl)
