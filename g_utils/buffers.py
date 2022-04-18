import numpy as np
import torch
from gym.spaces import Discrete, Box

from a_configuration.a_base_config.c_models.config_recurrent_convolutional_models import ConfigRecurrentConvolutionalModel
from a_configuration.a_base_config.c_models.config_recurrent_linear_models import ConfigRecurrentLinearModel
from g_utils.types import Transition


class Buffer:
    def __init__(self, action_space, config):
        self.action_space = action_space

        if isinstance(action_space, Discrete):
            self.n_out_actions = 1
        elif isinstance(action_space, Box):
            self.n_out_actions = action_space.shape[0]

        self.config = config
        self.is_recurrent_model = any([
            isinstance(self.config.MODEL_PARAMETER, ConfigRecurrentLinearModel),
            isinstance(self.config.MODEL_PARAMETER, ConfigRecurrentConvolutionalModel)
        ])
        self.internal_buffer = None
        self.size = None
        self.head = None

        self.clear()

    def clear(self):
        if self.config.BUFFER_CAPACITY > 0:
            self.internal_buffer = [None] * self.config.BUFFER_CAPACITY
            self.head = -1
        else:
            self.internal_buffer = []
            self.head = None

        self.size = 0

    def print_buffer(self):
        for idx, transition in enumerate(self.internal_buffer):
            print(f"Idx: {idx}, Transition: {transition}")
        print(f"Head: {self.head}, Size: {self.size}")

    def __len__(self):
        return self.size

    def append(self, transition):
        if self.config.BUFFER_CAPACITY > 0:
            self.head = (self.head + 1) % self.config.BUFFER_CAPACITY
            self.internal_buffer[self.head] = transition

            if self.size < self.config.BUFFER_CAPACITY:
                self.size += 1
        else:
            self.internal_buffer.append(transition)
            self.size += 1

    def sample_indices(self, batch_size):
        # Get index
        transition_indices = np.random.randint(self.size, size=batch_size)
        return transition_indices

    def sample(self, batch_size):
        if batch_size:
            if self.config.USE_PER:
                transition_indices, important_sampling_weights = self.sample_indices(batch_size)
            else:
                transition_indices = self.sample_indices(batch_size)

            # Sample
            observations, actions, next_observations, rewards, dones, infos = \
                zip(*[self.internal_buffer[idx] for idx in transition_indices])
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

            """
            type(hiddens): tuple
            len(hiddens): batch_size
            hiddens[0].shape: [num_layers]
            torch.stack(hiddens, 1).shape: [num_layers, batch_size, 1, hidden]
            torch.stack(hiddens, 1).squeeze(dim=2).shape: [num_layers, batch_size, hidden]
            """
            observations = torch.from_numpy(np.array(observations)).to(self.config.DEVICE)
            hiddens = torch.stack(hiddens, 1).squeeze(dim=2)

            next_observations = torch.from_numpy(np.array(next_observations)).to(self.config.DEVICE)
            next_hiddens = torch.stack(next_hiddens, 1).squeeze(dim=2)

            # if CNN
            if observations.ndim == 5:  # [batch_size, 1, channel, height, width]
                observations = observations.squeeze(1)
                next_observations = next_observations.squeeze(1)
                # [batch_size, channel, height, width]

            observations_v = [(observations, hiddens)]
            next_observations_v = [(next_observations, next_hiddens)]

        else:
            observations_v = torch.from_numpy(np.array(observations)).to(self.config.DEVICE)
            next_observations_v = torch.from_numpy(np.array(next_observations)).to(self.config.DEVICE)

        if isinstance(self.action_space, Discrete):
            # actions.shape = (256, 1)
            actions_v = torch.from_numpy(np.array(actions)).unsqueeze(dim=-1).to(self.config.DEVICE)

        elif isinstance(self.action_space, Box):
            # actions.shape = (256, 1)
            actions_v = torch.from_numpy(np.array(actions)).to(self.config.DEVICE)

        else:
            raise ValueError()

        rewards_v = torch.from_numpy(np.array(rewards, dtype=np.float32)).unsqueeze(dim=-1).to(self.config.DEVICE)
        dones_v = torch.from_numpy(np.array(dones, dtype=np.bool)).to(self.config.DEVICE)

        # print(observations_v.shape, actions_v.shape, next_observations_v.shape, rewards_v.shape, dones_v.shape)
        # observations.shape, next_observations.shape: (64, 4), (64, 4)
        # actions.shape, rewards.shape, dones.shape: (64, 1) (64, 1) (64,)

        # [MLP]
        # observations.shape: torch.Size([32, 4]),
        # actions.shape: torch.Size([32, 1]),
        # next_observations.shape: torch.Size([32, 4]),
        # rewards.shape: torch.Size([32, 1]),
        # dones.shape: torch.Size([32])
        #
        # [CNN]
        # observations.shape: torch.Size([32, 4, 84, 84]),
        # actions.shape: torch.Size([32, 1]),
        # next_observations.shape: torch.Size([32, 4, 84, 84]),
        # rewards.shape: torch.Size([32, 1]),
        # dones.shape: torch.Size([32])

        del observations
        del actions
        del next_observations
        del rewards
        del dones

        if self.config.USE_PER:
            return observations_v, actions_v, next_observations_v, rewards_v, dones_v, infos, important_sampling_weights
        else:
            return observations_v, actions_v, next_observations_v, rewards_v, dones_v, infos

    def sample_muzero(self, batch_size):
        """
        In muzero's buffer, transition is episode_history that save every step's
            observation
            action
            reward
            policy
            value
            gradient_scale
        """
        transition_indices, _ = self.sample_indices(batch_size)

        # Sample
        episode_idx, episode_history = zip(*[(idx, self.internal_buffer[idx]) for idx in transition_indices])
        return episode_idx, episode_history


if __name__ == "__main__":
    class Config:
        BUFFER_CAPACITY = 8
        MODEL_PARAMETER = None

    config = Config()

    buffer = Buffer(action_space=Discrete, config=config)
    buffer.print_buffer()

    for idx in range(config.BUFFER_CAPACITY + 4):
        buffer.append(Transition(
            observation=np.full((4,), idx),
            action=0,
            next_observation=np.full((4,), idx + 1),
            reward=1.0,
            done=False,
            info=None
        ))
        buffer.print_buffer()