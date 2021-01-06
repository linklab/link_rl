import random
import operator
import numpy as np

from codes.e_utils.experience import ExperienceSource, ExperienceSourceFirstLast
from codes.e_utils.experience_single import ExperienceSourceSingleEnvFirstLast


class ExperienceSourceBuffer:
    """
    The same as ExperienceSource, but takes episodes from the simple buffer
    """

    def __init__(self, buffer, steps_count=1):
        """
        Create buffered experience source
        :param buffer: list of episodes, each is a list of Experience object
        :param steps_count: count of steps in every entry
        """
        self.update_buffer(buffer)
        self.steps_count = steps_count

    def update_buffer(self, buffer):
        self.buffer = buffer
        self.lens = list(map(len, buffer))

    def __iter__(self):
        """
        Infinitely sample episode from the buffer and then sample item offset
        """
        while True:
            episode = random.randrange(len(self.buffer))
            ofs = random.randrange(self.lens[episode] - self.steps_count - 1)
            yield self.buffer[episode][ofs:ofs + self.steps_count]


class ExperienceReplayBuffer:
    def __init__(self, experience_source, buffer_size):
        assert isinstance(
            experience_source,
            (ExperienceSource, ExperienceSourceFirstLast, ExperienceSourceSingleEnvFirstLast, type(None))
        )
        assert isinstance(buffer_size, int)
        self.experience_source = experience_source
        self.experience_source_iter = None if experience_source is None else iter(experience_source)
        self.buffer = []
        self.capacity = buffer_size
        self.pos = 0

    def __len__(self):
        return len(self.buffer)

    def __iter__(self):
        return iter(self.buffer)

    def set_experience_source(self, experience_source):
        self.experience_source = experience_source
        self.experience_source_iter = None if experience_source is None else iter(experience_source)

    def sample(self, batch_size):
        """
        Get one random batch from experience replay
        TODO: implement sampling order policy
        :param batch_size:
        :return:
        """
        if len(self.buffer) <= batch_size:
            return self.buffer
        # Warning: replace=False makes random.choice O(n)
        keys = np.random.choice(len(self.buffer), batch_size, replace=True)
        return [self.buffer[key] for key in keys]

    def _add(self, sample):
        if len(self.buffer) < self.capacity:
            self.buffer.append(sample)
        else:
            self.buffer[self.pos] = sample
        self.pos = (self.pos + 1) % self.capacity

    def populate(self, num_samples):
        """
        Populates samples into the buffer
        :param samples: how many samples to populate
        """
        for _ in range(num_samples):
            entry = next(self.experience_source_iter)
            self._add(entry)

        return entry

    def populate_with_action_count(self, num_samples, action_count):
        """
        Populates samples into the buffer
        :param samples: how many samples to populate
        """
        for _ in range(num_samples):
            entry = next(self.experience_source_iter)
            action_count[entry.action] += 1
            self._add(entry)

    def populate_stacked_experience(self, num_samples, action_count=None):
        for _ in range(num_samples):
            exp = next(self.experience_source_iter)
            if action_count:
                action_count[exp.action] += 1
            # assert np.array_equal(exp.state.__array__()[1, :, :], exp.last_state.__array__()[0, :, :])
            # assert np.array_equal(exp.state.__array__()[2, :, :], exp.last_state.__array__()[1, :, :])
            # assert np.array_equal(exp.state.__array__()[3, :, :], exp.last_state.__array__()[2, :, :])

            extended_frames = np.zeros([5, 84, 84], dtype=np.uint8)
            extended_frames[0, :, :] = exp.state.__array__()[0, :, :]
            for i in range(1, 4):
                extended_frames[i, :, :] = exp.state.__array__()[i, :, :]

            if exp.last_state is not None:
                extended_frames[4, :, :] = exp.last_state.__array__()[3, :, :]

            self._add((extended_frames, exp.action, exp.reward, exp.last_state is None))

    def update_priorities(self, batch_indices, batch_priorities):
        raise NotImplementedError()

    def update_beta(self, idx):
        raise NotImplementedError()


class PrioReplayBufferNaive:
    def __init__(self, experience_source, buffer_size, prob_alpha=0.6):
        self.experience_source_iter = iter(experience_source)
        self.prob_alpha = prob_alpha
        self.capacity = buffer_size
        self.pos = 0
        self.buffer = []
        self.priorities = np.zeros((buffer_size,), dtype=np.float32)

    def __len__(self):
        return len(self.buffer)

    def set_experience_source(self, experience_source):
        self.experience_source = experience_source
        self.experience_source_iter = None if experience_source is None else iter(experience_source)
        
    def populate(self, count):
        max_prio = self.priorities.max() if self.buffer else 1.0
        for _ in range(count):
            sample = next(self.experience_source_iter)
            if len(self.buffer) < self.capacity:
                self.buffer.append(sample)
            else:
                self.buffer[self.pos] = sample
            self.priorities[self.pos] = max_prio
            self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]
        probs = np.array(prios, dtype=np.float32) ** self.prob_alpha

        probs /= probs.sum()
        indices = np.random.choice(len(self.buffer), batch_size, p=probs, replace=True)
        samples = [self.buffer[idx] for idx in indices]
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        return samples, indices, np.array(weights, dtype=np.float32)

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio


# sumtree 사용 버전
class PrioritizedReplayBuffer(ExperienceReplayBuffer):
    def __init__(self, experience_source, buffer_size, alpha=0.6, n_step=1, beta_start=0.4, beta_frames=100000):
        super(PrioritizedReplayBuffer, self).__init__(experience_source, buffer_size)
        assert alpha > 0
        self.alpha = alpha
        self.beta = beta_start
        self.n_step = n_step
        self.beta_start = beta_start
        self.beta_frames = beta_frames

        it_capacity = 1
        while it_capacity < buffer_size:
            it_capacity *= 2

        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 1.0

    def update_beta(self, idx):
        v = self.beta_start + idx * (1.0 - self.beta_start) / self.beta_frames
        self.beta = min(1.0, v)
        return self.beta

    def _add(self, *args, **kwargs):
        idx = self.pos
        super()._add(*args, **kwargs)
        self._it_sum[idx] = self._max_priority ** self.alpha
        self._it_min[idx] = self._max_priority ** self.alpha

    def _sample_proportional(self, batch_size):
        assert len(self) > self.n_step
        res = []
        for _ in range(batch_size):
            while True:
                mass = random.random() * self._it_sum.sum(0, len(self) - 1)
                idx = self._it_sum.find_prefixsum_idx(mass)

                upper = self.pos
                lower = (self.pos - self.n_step)
                if lower < 0:
                    lower = self.capacity + lower
                if lower < upper:
                    if not lower <= idx < upper:
                        res.append(idx)
                        break
                else:
                    if upper <= idx < lower:
                        res.append(idx)
                        break
        return res

    def sample(self, batch_size):
        assert self.beta > 0

        idxes = self._sample_proportional(batch_size)
        # print("#################")
        # print(idxes)
        # print(self.pos)
        # print("#################")

        weights = []
        p_min = self._it_min.min() / self._it_sum.sum()
        max_weight = (p_min * len(self)) ** (-self.beta)

        for idx in idxes:
            p_sample = self._it_sum[idx] / self._it_sum.sum()
            weight = (p_sample * len(self)) ** (-self.beta)
            weights.append(weight / max_weight)

        weights = np.array(weights, dtype=np.float32)
        samples = [self.buffer[idx] for idx in idxes]
        return samples, idxes, weights

    def update_priorities(self, idxes, priorities):
        # with torch.no_grad():
        assert len(idxes) == len(priorities)
        for idx, priority in zip(idxes, priorities):
            assert priority > 0.0, priority
            assert 0 <= idx < len(self), idx
            self._it_sum[idx] = priority ** self.alpha
            self._it_min[idx] = priority ** self.alpha

            self._max_priority = max(self._max_priority, priority)


class SegmentTree(object):
    def __init__(self, capacity, operation, neutral_element):
        """Build a Segment Tree data structure.

        https://en.wikipedia.org/wiki/Segment_tree

        Can be used as regular array, but with two
        important differences:

            a) setting item's value is slightly slower.
               It is O(lg capacity) instead of O(1).
            b) user has access to an efficient `reduce`
               operation which reduces `operation` over
               a contiguous subsequence of items in the
               array.

        Paramters
        ---------
        capacity: int
            Total size of the array - must be a power of two.
        operation: lambda obj, obj -> obj
            and operation for combining elements (eg. sum, max)
            must for a mathematical group together with the set of
            possible values for array elements.
        neutral_element: obj
            neutral element for the operation above. eg. float('-inf')
            for max and 0 for sum.
        """
        assert capacity > 0 and capacity & (capacity - 1) == 0, "capacity must be positive and a power of 2."
        self._capacity = capacity
        self._value = [neutral_element for _ in range(2 * capacity)]
        self._operation = operation

    def _reduce_helper(self, start, end, node, node_start, node_end):
        if start == node_start and end == node_end:
            return self._value[node]
        mid = (node_start + node_end) // 2
        if end <= mid:
            return self._reduce_helper(start, end, 2 * node, node_start, mid)
        else:
            if mid + 1 <= start:
                return self._reduce_helper(start, end, 2 * node + 1, mid + 1, node_end)
            else:
                return self._operation(
                    self._reduce_helper(start, mid, 2 * node, node_start, mid),
                    self._reduce_helper(mid + 1, end, 2 * node + 1, mid + 1, node_end)
                )

    def reduce(self, start=0, end=None):
        """Returns result of applying `self.operation`
        to a contiguous subsequence of the array.

            self.operation(arr[start], operation(arr[start+1], operation(... arr[end])))

        Parameters
        ----------
        start: int
            beginning of the subsequence
        end: int
            end of the subsequences

        Returns
        -------
        reduced: obj
            result of reducing self.operation over the specified range of array elements.
        """
        if end is None:
            end = self._capacity
        if end < 0:
            end += self._capacity
        end -= 1
        return self._reduce_helper(start, end, 1, 0, self._capacity - 1)

    def __setitem__(self, idx, val):
        # index of the leaf
        idx += self._capacity
        self._value[idx] = val
        idx //= 2
        while idx >= 1:
            self._value[idx] = self._operation(
                self._value[2 * idx],
                self._value[2 * idx + 1]
            )
            idx //= 2

    def __getitem__(self, idx):
        assert 0 <= idx < self._capacity
        return self._value[self._capacity + idx]


class SumSegmentTree(SegmentTree):
    def __init__(self, capacity):
        super(SumSegmentTree, self).__init__(
            capacity=capacity,
            operation=operator.add,
            neutral_element=0.0
        )

    def sum(self, start=0, end=None):
        """Returns arr[start] + ... + arr[end]"""
        return super(SumSegmentTree, self).reduce(start, end)

    def find_prefixsum_idx(self, prefixsum):
        """Find the highest index `i` in the array such that
            sum(arr[0] + arr[1] + ... + arr[i - i]) <= prefixsum

        if array values are probabilities, this function
        allows to sample indexes according to the discrete
        probability efficiently.

        Parameters
        ----------
        perfixsum: float
            upperbound on the sum of array prefix

        Returns
        -------
        idx: int
            highest index satisfying the prefixsum constraint
        """
        assert 0 <= prefixsum <= self.sum() + 1e-5
        idx = 1
        while idx < self._capacity:  # while non-leaf
            if self._value[2 * idx] > prefixsum:
                idx = 2 * idx
            else:
                prefixsum -= self._value[2 * idx]
                idx = 2 * idx + 1
        return idx - self._capacity


class MinSegmentTree(SegmentTree):
    def __init__(self, capacity):
        super(MinSegmentTree, self).__init__(
            capacity=capacity,
            operation=min,
            neutral_element=float('inf')
        )

    def min(self, start=0, end=None):
        """Returns min(arr[start], ...,  arr[end])"""

        return super(MinSegmentTree, self).reduce(start, end)