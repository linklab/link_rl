import collections
import numpy as np
import torch

from g_utils.types import Transition


class Buffer:
    def __init__(self, capacity, device=None):
        self.internal_buffer = collections.deque(maxlen=capacity)
        if device is None:
            self.device = torch.device("cpu")
        else:
            self.device = device

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
        return Buffer.sample_with_given_buffer(
            self.internal_buffer, batch_size, device=self.device
        )

    def sample_all_for_reinforce(self):
        _, _, _, observations, actions, _, rewards, _, _ = zip(*self.internal_buffer)

        # Convert to ndarray for speed up cuda
        observations = np.array(observations)
        # observations.shape: (64, 4)
        actions = np.array(actions)
        actions = np.expand_dims(actions, axis=-1) if actions.ndim == 1 else actions

        rewards = np.array(rewards)
        rewards = np.expand_dims(rewards, axis=-1) if rewards.ndim == 1 else rewards

        # Convert to tensor
        observations = torch.tensor(
            observations, dtype=torch.float32, device=self.device
        )

        actions = torch.tensor(actions, dtype=torch.int64, device=self.device)

        return observations, actions, rewards

    def get_filtered_buffer(self, model_version_v):
        filtered_buffer = [
            transition for transition in self.internal_buffer
            if transition.info["model_version_v"] == model_version_v
        ]

        # print(model_version_v, len(filtered_buffer), "!!!!!!")

        return filtered_buffer

    @staticmethod
    def sample_with_given_buffer(
            buffer, batch_size=None, device=torch.device("cpu")
    ):
        if batch_size:
            # Get index
            indices = np.random.choice(len(buffer), size=batch_size, replace=False)

            # Sample
            observations, actions, next_observations, rewards, dones, infos = \
                zip(*[buffer[idx] for idx in indices])
        else:
            observations, actions, next_observations, rewards, dones, infos = \
                zip(*buffer)

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
        observations = torch.tensor(
            observations, dtype=torch.float32, device=device
        )
        actions = torch.tensor(
            actions, dtype=torch.int64, device=device
        )
        next_observations = torch.tensor(
            next_observations, dtype=torch.float32, device=device
        )
        rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
        dones = torch.tensor(dones, dtype=torch.bool, device=device)

        return observations, actions, next_observations, rewards, dones
