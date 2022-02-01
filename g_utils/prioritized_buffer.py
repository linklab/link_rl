import numpy as np

from g_utils.buffers import Buffer


class SumTree:
    def __init__(self, config):
        self.config = config
        self.write = 0
        self.capacity = self.config.BUFFER_CAPACITY
        self.binary_tree = np.zeros(2 * self.capacity - 1)  # Stores the priorities and sums of priorities
        self.indices = np.zeros(self.capacity)              # Stores the indices of the experiences

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2

        self.binary_tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if len(self.binary_tree) <= left:
            return idx

        if s <= self.binary_tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.binary_tree[left])

    def total(self):
        return self.binary_tree[0]

    def add(self, p, index):
        idx = self.write + self.capacity - 1

        self.indices[self.write] = index
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

    def update(self, idx, p):
        change = p - self.binary_tree[idx]

        self.binary_tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        assert s <= self.total()
        idx = self._retrieve(0, s)
        indexIdx = idx - self.capacity + 1

        return (idx, self.binary_tree[idx], self.indices[indexIdx])

    def print_tree(self):
        for i in range(len(self.indices)):
            j = i + self.capacity - 1
            print(f'Idx: {i}, Data idx: {self.indices[i]}, Priority: {self.binary_tree[j]}')


class PrioritizedBuffer(Buffer):
    def __init__(self, action_space, config):
        super(PrioritizedBuffer, self).__init__(action_space, config)

        self.sum_tree = SumTree(self.config.BUFFER_CAPACITY)
        self.default_error = 100_000

    def clear(self):
        super(PrioritizedBuffer, self).clear()
        self.sum_tree = SumTree(self.config.BUFFER_CAPACITY)

    def append(self, transition):
        super(PrioritizedBuffer, self).append(transition)
        priority = self.get_priority(self.default_error)
        self.priorities[self.head] = priority
        self.tree.add(priority, self.head)

    def get_priority(self, error):
        '''Takes in the error of one or more examples and returns the proportional priority'''
        return np.power(error + self.config.PER_EPSILON, self.config.PER_ALPHA)
