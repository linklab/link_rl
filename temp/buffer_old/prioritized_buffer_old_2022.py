import math

import gym
import numpy as np
import random

from link_rl.h_utils.buffers.buffer import Buffer
from link_rl.h_utils.types import Transition


class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.binary_tree = np.zeros(2 * self.capacity - 1)            # Stores the priorities and sums of priorities
        self.transition_indices = np.zeros(self.capacity, dtype=int)  # Stores the indices of the experiences
        self.num_transitions = 0

    def _propagate(self, node_idx, change):
        parent_node_idx = (node_idx - 1) // 2

        self.binary_tree[parent_node_idx] += change

        if parent_node_idx != 0:
            self._propagate(parent_node_idx, change)

    def _retrieve(self, node_idx, x):
        left = 2 * node_idx + 1
        right = left + 1

        if len(self.binary_tree) <= left:
            return node_idx

        if x <= self.binary_tree[left]:
            return self._retrieve(left, x)
        else:
            return self._retrieve(right, x - self.binary_tree[left])

    def total(self):
        return self.binary_tree[0]

    def add(self, priority, head):
        node_idx = head + self.capacity - 1

        # print(f"ADD - Node_Idx: {node_idx}, Head: {head}, Priority: {priority}")

        self.transition_indices[head] = head
        self.update(node_idx, priority)

        self.num_transitions = min(self.num_transitions + 1, self.capacity)

    def update(self, node_idx, priority):
        change = priority - self.binary_tree[node_idx]

        self.binary_tree[node_idx] = priority
        self._propagate(node_idx, change)

    def get(self, x):
        node_idx = self._retrieve(0, x)
        head = node_idx - self.capacity + 1

        return node_idx, self.binary_tree[node_idx], head

    def print_tree(self):
        print("* SUN TREE *")
        print(f'Capacity: {self.capacity}')
        for j in range(self.capacity - 1):
            print(f'Idx: -, Transition idx: -, Node Idx: {j}, Priority: {self.binary_tree[j]}')

        for i in range(len(self.transition_indices)):
            j = i + self.capacity - 1
            print(f'Idx: {i}, Transition Idx: {self.transition_indices[i]}, Node Idx: {j}, Priority: {self.binary_tree[j]}')
        print(f"Total: {self.total()}")


class PrioritizedBuffer(Buffer):
    def __init__(self, observation_space, action_space, config):
        super(PrioritizedBuffer, self).__init__(observation_space, action_space, config)

        self.sum_tree = SumTree(self.config.BUFFER_CAPACITY)
        self.priorities = [None] * self.config.BUFFER_CAPACITY
        self.default_error = 100_000

        self.sampled_transition_indices = None
        self.sampled_node_indices = None

    def clear(self):
        super(PrioritizedBuffer, self).clear()
        self.sum_tree = SumTree(self.config.BUFFER_CAPACITY)
        self.priorities = [None] * self.config.BUFFER_CAPACITY

    def print_buffer(self):
        for idx, (transition, priority) in enumerate(zip(self.internal_buffer, self.priorities)):
            print(f"Internal Buffer Idx: {idx}, Transition: {transition}, Priority: {priority}")
        print(f"Head: {self.head}, Size: {self.size}")
        print()

    def append(self, transition):
        super(PrioritizedBuffer, self).append(transition)
        priority = self._get_priority(self.default_error)
        self.priorities[self.head] = priority
        self.sum_tree.add(priority, self.head)

    def _get_priority(self, td_error):
        '''
        - Takes in the td_error of one or more examples and returns the proportional priority
        - default_priority = (100_000 + 0.01)^0.6
        '''
        return np.power(td_error + self.config.PER_EPSILON, self.config.PER_ALPHA)

    def update_priorities(self, errors):
        '''
        Updates the priorities from the most recent batch
        Assumes the relevant batch indices are stored in self.batch_idxs
        '''
        priorities = self._get_priority(errors)
        assert len(priorities) == len(self.sampled_transition_indices)

        for idx, node_idx, priority in zip(self.sampled_transition_indices, self.sampled_node_indices, priorities):
            self.priorities[idx] = priority
            self.sum_tree.update(node_idx, priority)

    def sample_indices(self, batch_size):
        '''Samples batch_size indices from memory in proportional to their priority.'''
        transition_indices = np.zeros(batch_size, dtype=int)
        node_indices = np.zeros(batch_size, dtype=int)
        priorities = np.zeros(batch_size, dtype=float)

        xs = np.zeros(batch_size, dtype=float)

        for i in range(batch_size):
            x = random.uniform(0, self.sum_tree.total())
            xs[i] = x
            node_idx, priority, idx = self.sum_tree.get(x)

            #
            # if priority == 0.0:
            #     priority = self.config.PER_EPSILON

            transition_indices[i] = idx
            node_indices[i] = node_idx
            priorities[i] = priority

        self.sampled_transition_indices = np.asarray(transition_indices).astype(int)
        self.sampled_node_indices = node_indices

        priorities = np.asarray(priorities).astype(float)
        sampling_probabilities = priorities / self.sum_tree.total()

        important_sampling_weights = np.power(
            self.sum_tree.num_transitions * sampling_probabilities, -1.0 * self.config.PER_BETA
        )
        important_sampling_weights /= important_sampling_weights.max()

        if np.isnan(important_sampling_weights).any():
            nan_idx = None
            for idx, is_weight in enumerate(important_sampling_weights):
                if math.isnan(is_weight):
                    nan_idx = idx
                    break
            print(nan_idx, "!!!!!!!!!!")
            print(xs, xs[nan_idx], self.sum_tree.total(), self.sum_tree.num_transitions, "%%%% - 0")
            print(transition_indices, transition_indices[nan_idx], "%%%% - 1")
            print(node_indices, node_indices[nan_idx], "%%%% - 2")
            print(priorities, priorities[nan_idx], "%%%% - 3")
            print(important_sampling_weights, important_sampling_weights[nan_idx], "%%%% - 4")

        return transition_indices, important_sampling_weights


if __name__ == "__main__":
    class Config:
        BUFFER_CAPACITY = 4
        PER_ALPHA = 0.6
        PER_EPSILON = 0.01
        PER_BETA = 0.4
        MODEL_PARAMETER = None
        BATCH_SIZE = 2

    config = Config()

    observation_space = gym.spaces.Discrete(n=4)  # 0, 1, 2, 3
    action_space = gym.spaces.Discrete(n=3)  # 0, 1, 2

    prioritized_buffer = PrioritizedBuffer(observation_space=observation_space, action_space=action_space, config=config)
    print("#" * 100)
    prioritized_buffer.print_buffer()
    prioritized_buffer.sum_tree.print_tree()
    print("#" * 100);print()

    for idx in range(config.BUFFER_CAPACITY + 4):
        prioritized_buffer.append(Transition(
            observation=np.full((4,), idx),
            action=0,
            next_observation=np.full((4,), idx + 1),
            reward=1.0,
            done=False,
            info=None
        ))
        print("#" * 100)
        prioritized_buffer.print_buffer()
        prioritized_buffer.sum_tree.print_tree()
        print("#" * 100);print()

    print()

    print("SAMPLE & UPDATE #1")
    transition_indices, important_sampling_weights = prioritized_buffer.sample_indices(batch_size=config.BATCH_SIZE)
    print(transition_indices, important_sampling_weights)

    errors = np.ones_like(transition_indices)
    prioritized_buffer.update_priorities(errors)
    print("#" * 100)
    prioritized_buffer.sum_tree.print_tree()
    print("#" * 100);print()
    print()

    print("SAMPLE & UPDATE #2")
    transition_indices, important_sampling_weights = prioritized_buffer.sample_indices(batch_size=config.BATCH_SIZE)
    print(transition_indices, important_sampling_weights)

    errors = np.ones_like(transition_indices)
    prioritized_buffer.update_priorities(errors)
    print("#" * 100)
    prioritized_buffer.sum_tree.print_tree()
    print("#" * 100);print()
    print()

    print("SAMPLE & UPDATE #3")
    transition_indices, important_sampling_weights = prioritized_buffer.sample_indices(batch_size=config.BATCH_SIZE)
    print(transition_indices, important_sampling_weights)

    errors = np.full(transition_indices.shape, 100.0)
    prioritized_buffer.update_priorities(errors)
    print("#" * 100)
    prioritized_buffer.sum_tree.print_tree()
    print("#" * 100);print()
    print()

    print("SAMPLE & UPDATE #4")
    transition_indices, important_sampling_weights = prioritized_buffer.sample_indices(batch_size=config.BATCH_SIZE)
    print(transition_indices, important_sampling_weights)

    errors = np.full(transition_indices.shape, 10.0)
    prioritized_buffer.update_priorities(errors)
    print("#" * 100)
    prioritized_buffer.sum_tree.print_tree()
    print("#" * 100);print()
