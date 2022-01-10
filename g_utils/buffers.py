import collections
import numpy as np
import torch
from gym.spaces import Discrete, Box

from g_utils.types import Transition


class Buffer:
    def __init__(self, capacity, action_space, parameter):
        self.internal_buffer = collections.deque(maxlen=capacity)
        self.action_space = action_space
        self.parameter = parameter

    def __len__(self):
        return len(self.internal_buffer)

    def size(self):
        return len(self.internal_buffer)

    def pop(self):
        return self.internal_buffer.pop()

    def clear(self):
        self.internal_buffer.clear()

    def append(self, transition):
        self.internal_buffer.append(transition)

    def append_vectorized_transitions(self, transitions):
        for observation, action, next_observation, reward, done, info in zip(
                transitions.observations,
                transitions.actions,
                transitions.next_observations,
                transitions.rewards,
                transitions.dones,
                transitions.infos
        ):
            transition = Transition(
                observation=observation,
                action=action,
                next_observation=next_observation,
                reward=reward,
                done=done,
                info=info
            )
            self.internal_buffer.append(transition)

    def sample(self, batch_size):
        if batch_size:
            # Get index
            # indices = np.random.choice(len(self.internal_buffer), size=batch_size, replace=False)
            indices = np.random.randint(len(self.internal_buffer), size=batch_size)

            # Sample
            observations, actions, next_observations, rewards, dones, infos = \
                zip(*[self.internal_buffer[idx] for idx in indices])
        else:
            observations, actions, next_observations, rewards, dones, infos = \
                zip(*self.internal_buffer)

        # Convert to tensor
        observations_v = torch.tensor(observations, dtype=torch.float32, device=self.parameter.DEVICE)

        if isinstance(self.action_space, Discrete):     # actions.shape = (64,)
            actions_v = torch.tensor(actions, dtype=torch.int64, device=self.parameter.DEVICE)[:, None]
        elif isinstance(self.action_space, Box):        # actions.shape = (64, 8)
            actions_v = torch.tensor(actions, dtype=torch.int64, device=self.parameter.DEVICE)
        else:
            raise ValueError()

        next_observations_v = torch.tensor(next_observations, dtype=torch.float32, device=self.parameter.DEVICE)
        rewards_v = torch.tensor(rewards, dtype=torch.float32, device=self.parameter.DEVICE)[:, None]
        dones_v = torch.tensor(dones, dtype=torch.bool, device=self.parameter.DEVICE)

        # print(observations_v.shape, actions_v.shape, next_observations_v.shape, rewards_v.shape, dones_v.shape)
        # observations.shape, next_observations.shape: (64, 4), (64, 4)
        # actions.shape, rewards.shape, dones.shape: (64, 1) (64, 1) (64,)

        return observations_v, actions_v, next_observations_v, rewards_v, dones_v

    def sample_old(self, batch_size):
        if batch_size:
            # Get index
            indices = np.random.choice(len(self.internal_buffer), size=batch_size, replace=False)

            # Sample
            observations, actions, next_observations, rewards, dones, infos = \
                zip(*[self.internal_buffer[idx] for idx in indices])
        else:
            observations, actions, next_observations, rewards, dones, infos = \
                zip(*self.internal_buffer)

        # Convert to ndarray for speed up cuda
        observations = np.array(observations)
        next_observations = np.array(next_observations)
        # observations.shape, next_observations.shape: (64, 4), (64, 4)

        actions = np.array(actions)
        actions = np.expand_dims(actions, axis=-1) if actions.ndim == 1 else actions

        rewards = np.array(rewards)
        rewards = np.expand_dims(rewards, axis=-1) if rewards.ndim == 1 else rewards

        dones = np.array(dones, dtype=bool)
        # actions.shape, rewards.shape, dones.shape: (64, 1) (64, 1) (64,)

        # Convert to tensor
        observations_v = torch.tensor(observations, dtype=torch.float32, device=self.parameter.DEVICE)
        actions_v = torch.tensor(actions, dtype=torch.int64, device=self.parameter.DEVICE)
        next_observations_v = torch.tensor(next_observations, dtype=torch.float32, device=self.parameter.DEVICE)
        rewards_v = torch.tensor([rewards], dtype=torch.float32, device=self.parameter.DEVICE)
        dones_v = torch.tensor(dones, dtype=torch.bool, device=self.parameter.DEVICE)

        del observations
        del actions
        del next_observations
        del rewards
        del dones

        return observations_v, actions_v, next_observations_v, rewards_v, dones_v

