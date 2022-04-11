import math

import numpy as np
import random

from gym.spaces import Discrete

from g_utils.buffers import Buffer
from g_utils.types import Transition


class SumTree:
    def __init__(self, capacity):
        self.write = 0
        self.capacity = capacity
        self.binary_tree = np.zeros(2 * self.capacity - 1)            # Stores the priorities and sums of priorities
        self.transition_indices = np.zeros(self.capacity, dtype=int)  # Stores the indices of the experiences

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

    def add(self, priority, index):
        node_idx = self.write + self.capacity - 1

        self.transition_indices[self.write] = index
        self.update(node_idx, priority)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

    def update(self, node_idx, priority):
        change = priority - self.binary_tree[node_idx]

        self.binary_tree[node_idx] = priority
        self._propagate(node_idx, change)

    def get(self, x):
        assert x <= self.total(), "{0} {1}".format(x, self.total())
        node_idx = self._retrieve(0, x)
        index_idx = node_idx - self.capacity + 1

        return node_idx, self.binary_tree[node_idx], self.transition_indices[index_idx]

    def num_transition_entries(self):
        return len(self.transition_indices)

    def print_tree(self):
        print("#" * 100)
        print(f'Capacity: {self.capacity}')
        for j in range(self.capacity - 1):
            print(f'Idx: -, Data idx: -, Node Idx: {j}, Priority: {self.binary_tree[j]}')

        for i in range(len(self.transition_indices)):
            j = i + self.capacity - 1
            print(f'Idx: {i}, Data idx: {self.transition_indices[i]}, Node Idx: {j}, Priority: {self.binary_tree[j]}')
        print(f"Total: {self.total()}")
        print("#" * 100)


class PrioritizedBuffer(Buffer):
    def __init__(self, action_space, config):
        super(PrioritizedBuffer, self).__init__(action_space, config)

        self.sum_tree = SumTree(self.config.BUFFER_CAPACITY)
        self.priorities = [None] * self.config.BUFFER_CAPACITY
        self.default_error = 100_000

        self.sampled_transition_indices = None
        self.sampled_node_indices = None

    def clear(self):
        super(PrioritizedBuffer, self).clear()
        self.sum_tree = SumTree(self.config.BUFFER_CAPACITY)
        self.priorities = [None] * self.config.BUFFER_CAPACITY

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

    def sample_indices(self, batch_size):
        '''Samples batch_size indices from memory in proportional to their priority.'''
        transition_indices = np.zeros(batch_size, dtype=int)
        node_indices = np.zeros(batch_size, dtype=int)
        priorities = np.zeros(batch_size, dtype=float)

        for i in range(batch_size):
            x = random.uniform(0, self.sum_tree.total())
            node_idx, priority, idx = self.sum_tree.get(x)
            transition_indices[i] = idx
            node_indices[i] = node_idx
            priorities[i] = priority

        self.sampled_transition_indices = np.asarray(transition_indices).astype(int)
        self.sampled_node_indices = node_indices

        priorities = np.asarray(priorities).astype(float)
        sampling_probabilities = priorities / self.sum_tree.total()

        important_sampling_weights = np.power(
            self.sum_tree.num_transition_entries() * sampling_probabilities,
            -1.0 * self.config.PER_BETA
        )
        important_sampling_weights /= important_sampling_weights.max()

        return transition_indices, important_sampling_weights

    def update_priorities(self, errors):
        '''
        Updates the priorities from the most recent batch
        Assumes the relevant batch indices are stored in self.batch_idxs
        '''
        priorities = self._get_priority(errors)
        assert len(priorities) == len(self.sampled_transition_indices)

        for idx, node_idx, priority in zip(self.sampled_transition_indices, self.sampled_node_indices, priorities):
            if math.isnan(priority):
                continue
            assert not math.isnan(priority), "{0} {1} {2}".format(idx, node_idx, priority)
            self.priorities[idx] = priority
            self.sum_tree.update(node_idx, priority)


if __name__ == "__main__":
    class Config:
        BUFFER_CAPACITY = 6
        PER_ALPHA = 0.6
        PER_EPSILON = 0.01
        MODEL_PARAMETER = None

    config = Config()

    prioritized_buffer = PrioritizedBuffer(action_space=Discrete, config=config)
    prioritized_buffer.append(Transition(
        observation=np.full((4,), 0.0), action=0, next_observation=np.full((4,), 1.0), reward=1.0, done=False, info=None
    ))
    prioritized_buffer.append(Transition(
        observation=np.full((4,), 0.0), action=0, next_observation=np.full((4,), 1.0), reward=1.0, done=False, info=None
    ))
    prioritized_buffer.append(Transition(
        observation=np.full((4,), 0.0), action=0, next_observation=np.full((4,), 1.0), reward=1.0, done=False, info=None
    ))
    prioritized_buffer.append(Transition(
        observation=np.full((4,), 0.0), action=0, next_observation=np.full((4,), 1.0), reward=1.0, done=False, info=None
    ))
    prioritized_buffer.append(Transition(
        observation=np.full((4,), 0.0), action=0, next_observation=np.full((4,), 1.0), reward=1.0, done=False, info=None
    ))
    prioritized_buffer.append(Transition(
        observation=np.full((4,), 0.0), action=0, next_observation=np.full((4,), 1.0), reward=1.0, done=False, info=None
    ))
    print("INITIALIZE")
    prioritized_buffer.sum_tree.print_tree()

    print()

    print("SAMPLE & UPDATE #1")
    samples, priorities = prioritized_buffer.sample_indices(batch_size=3)
    print(samples, priorities)

    errors = np.ones_like(samples)
    prioritized_buffer.update_priorities(errors)
    prioritized_buffer.sum_tree.print_tree()

    print()

    print("SAMPLE & UPDATE #2")
    samples, priorities = prioritized_buffer.sample_indices(batch_size=3)
    print(samples, priorities)

    errors = np.ones_like(samples)
    prioritized_buffer.update_priorities(errors)
    prioritized_buffer.sum_tree.print_tree()

    print()

    print("SAMPLE & UPDATE #3")
    samples, priorities = prioritized_buffer.sample_indices(batch_size=3)
    print(samples, priorities)

    errors = np.full(samples.shape, 100.0)
    prioritized_buffer.update_priorities(errors)
    prioritized_buffer.sum_tree.print_tree()

    print()

    print("SAMPLE & UPDATE #4")
    samples, priorities = prioritized_buffer.sample_indices(batch_size=3)
    print(samples, priorities)

    errors = np.full(samples.shape, 10.0)
    prioritized_buffer.update_priorities(errors)
    prioritized_buffer.sum_tree.print_tree()