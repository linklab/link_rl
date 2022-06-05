import collections
import numpy as np
import torch
from gym.spaces import Discrete, Box

from a_configuration.a_base_config.c_models.config_recurrent_convolutional_models import \
    ConfigRecurrent2DConvolutionalModel, ConfigRecurrent1DConvolutionalModel
from a_configuration.a_base_config.c_models.config_recurrent_linear_models import ConfigRecurrentLinearModel
from g_utils.types import Transition


class Buffer:
    def __init__(self, action_space, config):
        self.action_space = action_space
        self.config = config
        self.internal_buffer = collections.deque(maxlen=self.config.BUFFER_CAPACITY)

        self.is_recurrent_model = any([
            isinstance(self.config.MODEL_PARAMETER, ConfigRecurrentLinearModel),
            isinstance(self.config.MODEL_PARAMETER, ConfigRecurrent1DConvolutionalModel),
            isinstance(self.config.MODEL_PARAMETER, ConfigRecurrent2DConvolutionalModel)
        ])

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

    def sample(self, batch_size):
        if batch_size:
            # Get index
            # indices = np.random.choice(len(self.internal_buffer), size=batch_size, replace=False)
            indices = np.random.randint(len(self.internal_buffer), size=batch_size)

            # Sample
            observations, actions, next_observations, rewards, dones, infos = \
                zip(*[self.internal_buffer[idx] for idx in indices])
        else:
            observations, actions, next_observations, rewards, dones, infos = zip(*self.internal_buffer)

        if self.is_recurrent_model:
            """
            type(observations): tuple ot tuple
            observations.shape: (batch_size, 2)
            
            observations, hiddens = zip(*observations)
            len(observations): batch_size
            len(hiddens): batch_size
            """
            observations, hiddens = zip(*observations)
            next_observations, next_hiddens = zip(*next_observations)

        # Convert to tensor
        if self.is_recurrent_model:
            """
            type(hiddens): tuple
            len(hiddens): batch_size
            hiddens[0].shape: [num_layers]
            torch.stack(hiddens, 1).shape: [num_layers, batch_size, 1, hidden]
            torch.stack(hiddens, 1).squeeze(dim=2).shape: [num_layers, batch_size, hidden]
            """
            observations = torch.tensor(observations, dtype=torch.float32, device=self.config.DEVICE)
            hiddens = torch.stack(hiddens, 1).squeeze(dim=2)

            next_observations = torch.tensor(next_observations, dtype=torch.float32, device=self.config.DEVICE)
            next_hiddens = torch.stack(next_hiddens, 1).squeeze(dim=2)

            # if CNN
            if observations.ndim == 5:  # [batch_size, 1, channel, height, width]
                observations = observations.squeeze(1)
                next_observations = next_observations.squeeze(1)
                # [batch_size, channel, height, width]

            observations_v = [(observations, hiddens)]
            next_observations_v = [(next_observations, next_hiddens)]

        else:
            observations_v = torch.tensor(observations, dtype=torch.float32, device=self.config.DEVICE)
            next_observations_v = torch.tensor(next_observations, dtype=torch.float32, device=self.config.DEVICE)

        if isinstance(self.action_space, Discrete):     # actions.shape = (64,)
            actions_v = torch.tensor(actions, dtype=torch.int64, device=self.config.DEVICE)[:, None]
        elif isinstance(self.action_space, Box):        # actions.shape = (64, 8)
            actions_v = torch.tensor(actions, dtype=torch.int64, device=self.config.DEVICE)
        else:
            raise ValueError()

        rewards_v = torch.tensor(rewards, dtype=torch.float32, device=self.config.DEVICE)[:, None]
        dones_v = torch.tensor(dones, dtype=torch.bool, device=self.config.DEVICE)

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
        observations_v = torch.tensor(observations, dtype=torch.float32, device=self.config.DEVICE)
        actions_v = torch.tensor(actions, dtype=torch.int64, device=self.config.DEVICE)
        next_observations_v = torch.tensor(next_observations, dtype=torch.float32, device=self.config.DEVICE)
        rewards_v = torch.tensor([rewards], dtype=torch.float32, device=self.config.DEVICE)
        dones_v = torch.tensor(dones, dtype=torch.bool, device=self.config.DEVICE)

        del observations
        del actions
        del next_observations
        del rewards
        del dones

        return observations_v, actions_v, next_observations_v, rewards_v, dones_v

