import gym
import numpy as np
import torch
from gym.spaces import Discrete, Box

from link_rl.a_configuration.a_base_config.c_models.config_recurrent_convolutional_models import \
    ConfigRecurrent2DConvolutionalModel, ConfigRecurrent1DConvolutionalModel
from link_rl.a_configuration.a_base_config.c_models.config_recurrent_linear_models import ConfigRecurrentLinearModel
from link_rl.g_utils.types import Transition


class Buffer:
    def __init__(self, observation_space, action_space, config):
        self.observation_space = observation_space
        self.action_space = action_space

        if isinstance(action_space, Discrete):
            self.n_out_actions = 1
        elif isinstance(action_space, Box):
            self.n_out_actions = action_space.shape[0]
        else:
            raise ValueError()

        self.config = config
        self.is_recurrent_model = any([
            isinstance(self.config.MODEL_PARAMETER, ConfigRecurrentLinearModel),
            isinstance(self.config.MODEL_PARAMETER, ConfigRecurrent1DConvolutionalModel),
            isinstance(self.config.MODEL_PARAMETER, ConfigRecurrent2DConvolutionalModel)
        ])
        self.internal_buffer = None

        self.observations_buffer = None
        self.actions_buffer = None
        self.next_observation_value = None
        self.rewards_buffer = None
        self.dones_buffer = None
        self.infos_buffer = None

        self.size = None
        self.head = None

        self.clear()

    @staticmethod
    def get_new_buffer_without_capacity(observation_space, action_space, config):
        observations_buffer = torch.zeros(size=(0, *observation_space.shape), dtype=torch.float16, device=config.DEVICE)

        if isinstance(action_space, Discrete):
            actions_buffer = torch.zeros(size=(0,), dtype=torch.int64, device=config.DEVICE)
        elif isinstance(action_space, Box):
            actions_buffer = torch.zeros(size=(0, *action_space.shape), device=config.DEVICE)
        else:
            raise ValueError()

        rewards_buffer = torch.zeros(size=(0,), device=config.DEVICE)
        dones_buffer = torch.zeros(size=(0,), dtype=torch.bool, device=config.DEVICE)
        infos_buffer = [None] * config.BUFFER_CAPACITY
        return observations_buffer, actions_buffer, rewards_buffer, dones_buffer, infos_buffer

    @staticmethod
    def get_new_buffer_with_capacity(observation_space, action_space, config):
        observations_buffer = torch.zeros(size=(config.BUFFER_CAPACITY, *observation_space.shape), dtype=torch.float16,
                                          device=config.DEVICE)

        if isinstance(action_space, Discrete):
            actions_buffer = torch.zeros(size=(config.BUFFER_CAPACITY,), dtype=torch.int64, device=config.DEVICE)
        elif isinstance(action_space, Box):
            actions_buffer = torch.zeros(size=(config.BUFFER_CAPACITY, *action_space.shape), device=config.DEVICE)
        else:
            raise ValueError()

        rewards_buffer = torch.zeros(size=(config.BUFFER_CAPACITY,), device=config.DEVICE)
        dones_buffer = torch.zeros(size=(config.BUFFER_CAPACITY,), dtype=torch.bool, device=config.DEVICE)
        infos_buffer = [None] * config.BUFFER_CAPACITY
        return observations_buffer, actions_buffer, rewards_buffer, dones_buffer, infos_buffer

    def clear(self):
        if self.config.BUFFER_CAPACITY > 0:
            self.observations_buffer, \
            self.actions_buffer, \
            self.rewards_buffer, \
            self.dones_buffer, \
            self.infos_buffer = Buffer.get_new_buffer_with_capacity(
                observation_space=self.observation_space, action_space=self.action_space, config=self.config
            )

            self.head = -1
        else:
            self.observations_buffer, \
            self.actions_buffer, \
            self.rewards_buffer, \
            self.dones_buffer, \
            self.infos_buffer = Buffer.get_new_buffer_without_capacity(
                observation_space=self.observation_space, action_space=self.action_space, config=self.config
            )

            self.head = None

        self.size = 0

    def print_buffer(self):
        transitions_in_buffer = zip(
            self.observations_buffer,
            self.actions_buffer,
            self.rewards_buffer,
            self.dones_buffer
        )
        for idx, transition in enumerate(transitions_in_buffer):
            print(f"Idx: {idx}, Transition: {transition}")

        print(f"Head: {self.head}, Size: {self.size}, \n"
              f"observations_buffer.shape: {self.observations_buffer.shape}, \n"
              f"actions_buffer.shape: {self.actions_buffer.shape}, \n"
              f"rewards_buffer.shape: {self.rewards_buffer.shape}, \n"
              f"dones_buffer.shape: {self.dones_buffer.shape}"
              )
        print()

    def __len__(self):
        return self.size

    def append(self, transition):
        if self.config.BUFFER_CAPACITY > 0:
            self.head = (self.head + 1) % self.config.BUFFER_CAPACITY

            self.observations_buffer[self.head] = torch.from_numpy(transition.observation).to(
                self.config.DEVICE)  # TENSOR

            if isinstance(self.action_space, Discrete):
                self.actions_buffer[self.head] = transition.action
            elif isinstance(self.action_space, Box):
                self.actions_buffer[self.head] = torch.from_numpy(transition.action).to(self.config.DEVICE)  # TENSOR
            else:
                raise ValueError()

            self.next_observation_value = torch.from_numpy(transition.next_observation).to(
                self.config.DEVICE)  # TENSOR
            self.rewards_buffer[self.head] = transition.reward
            self.dones_buffer[self.head] = bool(transition.done)
            self.infos_buffer[self.head] = transition.info

            if self.size < self.config.BUFFER_CAPACITY:
                self.size += 1
        else:
            self.observations_buffer = torch.cat(  # TENSOR
                (self.observations_buffer,
                 torch.unsqueeze(torch.from_numpy(transition.observation).to(self.config.DEVICE), dim=0)), dim=0
            )

            if isinstance(self.action_space, Discrete):
                self.actions_buffer = torch.cat(
                    (self.actions_buffer, torch.full((1,), fill_value=transition.action, device=self.config.DEVICE)),
                    dim=0
                )
            elif isinstance(self.action_space, Box):
                self.actions_buffer = torch.cat(
                    (self.actions_buffer,
                     torch.unsqueeze(torch.from_numpy(transition.action).to(self.config.DEVICE), dim=0)), dim=0
                    # TENSOR
                )
            else:
                raise ValueError()

            self.next_observation_value = torch.from_numpy(transition.next_observation).to(
                self.config.DEVICE)  # TENSOR

            self.rewards_buffer = torch.cat(
                (self.rewards_buffer, torch.full((1,), fill_value=transition.reward, device=self.config.DEVICE)), dim=0
            )

            self.dones_buffer = torch.cat(
                (self.dones_buffer, torch.full((1,), fill_value=bool(transition.done), device=self.config.DEVICE)),
                dim=0
            )

            self.infos_buffer.append(transition.info)

            self.size += 1

    def sample_indices(self, batch_size):
        # Get index
        transition_indices = np.random.randint(self.size, size=batch_size)
        return transition_indices

    def sample(self, batch_size):
        assert self.config.N_VECTORIZED_ENVS == 1

        if batch_size:
            if self.config.USE_PER:
                transition_indices, important_sampling_weights = self.sample_indices(batch_size)
            else:
                transition_indices = self.sample_indices(batch_size)

            observations = self.observations_buffer[transition_indices].to(torch.float32)

            if isinstance(self.action_space, Discrete):
                actions = self.actions_buffer[transition_indices].unsqueeze(dim=-1)
            elif isinstance(self.action_space, Box):
                actions = self.actions_buffer[transition_indices]
            else:
                raise ValueError()

            rewards = self.rewards_buffer[transition_indices].unsqueeze(dim=-1)
            dones = self.dones_buffer[transition_indices]
            infos = [self.infos_buffer[idx] for idx in transition_indices]

            next_observation_value_indices = np.where(transition_indices == self.head, -1, transition_indices)
            next_observation_value_indices = torch.tensor(next_observation_value_indices).to(self.config.DEVICE)

            observation_ndim = observations.ndim - 1
            observation_view = [1] * observation_ndim
            next_observation_value_indices = next_observation_value_indices.view(len(next_observation_value_indices), *observation_view)

            next_observations = self.observations_buffer[(transition_indices + 1) % self.config.BUFFER_CAPACITY].to(
                torch.float32)
            next_observations = torch.where(next_observation_value_indices < 0,
                                            self.next_observation_value.to(torch.float32), next_observations)

            # print(observations, observations.shape)
            # print(actions, actions.shape)
            # print(next_observations, next_observations.shape)
            # print(rewards, rewards.shape)
            # print(dones, dones.shape)
            # print(infos)
        else:
            observations = self.observations_buffer.to(torch.float32)

            if isinstance(self.action_space, Discrete):
                actions = self.actions_buffer.unsqueeze(dim=-1)
            elif isinstance(self.action_space, Box):
                actions = self.actions_buffer
            else:
                raise ValueError()

            next_observations = self.observations_buffer.clone().detach().to(torch.float32)

            next_observations = torch.cat(  # TENSOR
                (next_observations,
                 torch.unsqueeze(torch.from_numpy(self.next_observation_value.numpy()).to(self.config.DEVICE), dim=0)),
                dim=0
            )

            next_observations = next_observations[1:]

            rewards = self.rewards_buffer.unsqueeze(dim=-1)
            dones = self.dones_buffer
            infos = self.infos_buffer

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
            hiddens[0].shape: [num_layers, 1, hidden], 1 is the number of envs
            torch.stack(hiddens, 1).shape: [num_layers, batch_size, 1, hidden]
            torch.stack(hiddens, 1).squeeze(dim=2).shape: [num_layers, batch_size, hidden]
            """
            # print(observations[0].shape)
            # print(hiddens[0].shape)
            observations = torch.from_numpy(np.array(observations, dtype=np.float32)).to(self.config.DEVICE)
            hiddens = torch.stack(tensors=hiddens, dim=1).squeeze(dim=2)

            next_observations = torch.from_numpy(np.array(next_observations, dtype=np.float32)).to(self.config.DEVICE)
            next_hiddens = torch.stack(tensors=next_hiddens, dim=1).squeeze(dim=2)

            # if CNN
            if observations.ndim == 5:  # [batch_size, 1, channel, height, width]
                observations = observations.squeeze(1)
                next_observations = next_observations.squeeze(1)
                # [batch_size, channel, height, width]

            observations = [(observations, hiddens)]
            next_observations = [(next_observations, next_hiddens)]

        else:
            # observations_v = torch.from_numpy(np.array(observations, dtype=np.float32)).to(self.config.DEVICE)
            # next_observations_v = torch.from_numpy(np.array(next_observations, dtype=np.float32)).to(self.config.DEVICE)
            pass

        if self.config.USE_PER:
            return observations, actions, next_observations, rewards, dones, infos, important_sampling_weights
        else:
            return observations, actions, next_observations, rewards, dones, infos


def buffer_sample_test(buffer):
    print("SAMPLE")

    observations_v, actions_v, next_observations_v, rewards_v, dones_v, infos = buffer.sample(batch_size=4)


def buffer_test(config):
    observation_space = gym.spaces.Box(
        low=np.float32([-10, -10, -10, -10]),  # shape: (4,)
        high=np.float32([10, 10, 10, 10])  # shape: (4,)
    )
    action_space = gym.spaces.Discrete(n=3)  # 0, 1, 2

    buffer = Buffer(observation_space=observation_space, action_space=action_space, config=config)
    buffer.print_buffer()

    for idx in range(config.BUFFER_CAPACITY + 4):
        buffer.append(Transition(
            observation=torch.full(size=(4,), fill_value=idx),
            action=100,
            next_observation=torch.full(size=(4,), fill_value=idx + 1),
            reward=1.0,
            done=False,
            info=None
        ))
        buffer.print_buffer()

    return buffer


def buffer_test_with_continuous_actions(config):
    observation_space = gym.spaces.Box(
        low=np.float32([-10, -10, -10, -10]),  # shape: (4,)
        high=np.float32([10, 10, 10, 10])  # shape: (4,)
    )
    action_space = gym.spaces.Box(
        low=np.float32([-1, -1]),  # shape: (2,)
        high=np.float32([1, 1])  # shape: (2,)
    )

    buffer = Buffer(observation_space=observation_space, action_space=action_space, config=config)
    buffer.print_buffer()

    for idx in range(config.BUFFER_CAPACITY + 4):
        buffer.append(Transition(
            observation=torch.full(size=(4,), fill_value=idx),
            action=torch.full(size=(2,), fill_value=0.01),
            next_observation=torch.full(size=(4,), fill_value=idx + 1),
            reward=1.0,
            done=False,
            info=None
        ))
        buffer.print_buffer()

    return buffer


def cnn_buffer_test(config):
    observation_space = gym.spaces.Box(
        low=np.float32([[[-10, -10], [-10, -10]], [[-10, -10], [-10, -10]], [[-10, -10], [-10, -10]]]),
        # shape: 3, 2, 2
        high=np.float32([[[10, 10], [10, 10]], [[10, -10], [10, 10]], [[10, -10], [10, 10]]])  # shape: 3, 2, 2
    )
    action_space = gym.spaces.Discrete(n=3)  # 0, 1, 2

    buffer = Buffer(observation_space=observation_space, action_space=action_space, config=config)
    buffer.print_buffer()

    for idx in range(config.BUFFER_CAPACITY + 4):
        buffer.append(Transition(
            observation=torch.full(size=(3, 2, 2), fill_value=idx),
            action=100,
            next_observation=torch.full(size=(3, 2, 2), fill_value=idx + 1),
            reward=1.0,
            done=False,
            info=None
        ))
        buffer.print_buffer()

    return buffer


def cnn_buffer_test_with_continuous_actions(config):
    observation_space = gym.spaces.Box(
        low=np.float32([[[-10, -10], [-10, -10]], [[-10, -10], [-10, -10]], [[-10, -10], [-10, -10]]]),
        # shape: 3, 2, 2
        high=np.float32([[[10, 10], [10, 10]], [[10, -10], [10, 10]], [[10, -10], [10, 10]]])  # shape: 3, 2, 2
    )
    action_space = gym.spaces.Box(
        low=np.float32([-1, -1]),  # shape: (2,)
        high=np.float32([1, 1])  # shape: (2,)
    )

    buffer = Buffer(observation_space=observation_space, action_space=action_space, config=config)
    buffer.print_buffer()

    for idx in range(config.BUFFER_CAPACITY + 4):
        buffer.append(Transition(
            observation=torch.full(size=(3, 2, 2), fill_value=idx),
            action=torch.full(size=(2,), fill_value=0.01),
            next_observation=torch.full(size=(3, 2, 2), fill_value=idx + 1),
            reward=1.0,
            done=False,
            info=None
        ))
        buffer.print_buffer()

    return buffer


if __name__ == "__main__":
    class Config:
        BUFFER_CAPACITY = 8
        MODEL_PARAMETER = None
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        USE_PER = False
        N_VECTORIZED_ENVS = 1


    config = Config()

    buffer = buffer_test(config)
    # buffer = buffer_test_with_continuous_actions(config)
    # buffer = cnn_buffer_test(config)
    # buffer = cnn_buffer_test_with_continuous_actions(config)

    buffer_sample_test(buffer)