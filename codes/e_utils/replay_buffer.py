import math
import random
import operator
import sys

import numpy as np

from codes.e_utils.experience import ExperienceSource, ExperienceSourceFirstLast
from codes.e_utils.experience_single import ExperienceSourceSingleEnvFirstLast


class TrajectoryBuffer:
    def __init__(self, experience_source):
        self.experience_source = experience_source
        self.experience_source_iter = None if experience_source is None else iter(experience_source)
        self.buffer = []

    def __len__(self):
        return len(self.buffer)

    def set_experience_source(self, experience_source):
        self.experience_source = experience_source
        self.experience_source_iter = None if experience_source is None else iter(experience_source)

    def populate(self, num_samples):
        entry = None
        for _ in range(num_samples):
            entry = next(self.experience_source_iter)
            self.buffer.append(entry)

        return entry

    def sample_all(self):
        return self.buffer

    def clear(self):
        self.buffer.clear()


class ExperienceSourceBuffer:
    """
    The same as ExperienceSource, but takes episodes from the simple buffer
    """

    def __init__(self, buffer, n_step=1):
        """
        Create buffered experience source
        :param buffer: list of episodes, each is a list of Experience object
        :param n_step: count of steps in every entry
        """
        self.update_buffer(buffer)
        self.n_step = n_step

    def update_buffer(self, buffer):
        self.buffer = buffer
        self.lens = list(map(len, buffer))

    def __iter__(self):
        """
        Infinitely sample episode from the buffer and then sample item offset
        """
        while True:
            episode = random.randrange(len(self.buffer))
            ofs = random.randrange(self.lens[episode] - self.n_step - 1)
            yield self.buffer[episode][ofs:ofs + self.n_step]


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
        if batch_size is None or len(self.buffer) <= batch_size:
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
        entry = None
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

    def update_priorities(self, batch_indices, batch_priorities):
        raise NotImplementedError()

    def update_beta(self, idx):
        raise NotImplementedError()

    def clear(self):
        self.buffer.clear()
        self.pos = 0

    def size(self):
        return len(self.buffer)


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

    def size(self):
        return len(self.buffer)


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
        self.buffer_size = buffer_size

        it_capacity = 1
        while it_capacity < self.buffer_size:
            it_capacity *= 2

        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 1.0

    def clear(self):
        self.buffer.clear()

        it_capacity = 1
        while it_capacity < self.buffer_size:
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


class RankBasedPrioritizedReplayBuffer(ExperienceReplayBuffer):
    def __init__(self, experience_source, buffer_size, params, alpha=0.7, beta_start=0.5, beta_frames=100000):
        super(RankBasedPrioritizedReplayBuffer, self).__init__(experience_source, buffer_size)
        assert alpha > 0
        self.alpha = alpha
        self.beta = beta_start
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self._max_priority = 1.0

        self.replace_flag = True
        self.learn_start = params.MIN_REPLAY_SIZE_FOR_TRAIN
        self.total_steps = params.MAX_GLOBAL_STEP
        # partition number N, split total size to N part
        self.partition_num = 100
        self.batch_size = params.BATCH_SIZE

        self.index = 0
        self.record_size = 0
        self.isFull = False

        self.buffer = {}
        # self._experience = {}
        self.priority_queue = BinaryHeap(self.capacity)
        self.distributions = self.build_distributions()

        # self.beta_grad = (1 - self.beta_start) / (self.total_steps - self.learn_start)

    def build_distributions(self):
        """
        preprocess pow of rank
        (rank i) ^ (-alpha) / sum ((rank i) ^ (-alpha))
        :return: distributions, dict
        """
        res = {}
        n_partitions = self.partition_num
        partition_num = 1
        # each part size
        partition_size = int(math.floor(1.0*self.capacity / n_partitions))

        for n in range(partition_size, self.capacity + 1, partition_size):
            if self.learn_start <= n <= self.capacity:
                distribution = {}
                # P(i) = (rank i) ^ (-alpha) / sum ((rank i) ^ (-alpha))
                pdf = list(
                    map(lambda x: math.pow(x, -self.alpha), range(1, n + 1))
                )
                pdf_sum = math.fsum(pdf)
                distribution['pdf'] = list(map(lambda x: x / pdf_sum, pdf))
                # split to k segment, and than uniform sample in each k
                # set k = batch_size, each segment has total probability is 1 / batch_size
                # strata_ends keep each segment start pos and end pos
                cdf = np.cumsum(distribution['pdf'])
                strata_ends = {1: 0, self.batch_size + 1: n}
                step = 1.0 / self.batch_size
                index = 1
                for s in range(2, self.batch_size + 1):
                    while cdf[index] < step:
                        index += 1
                    strata_ends[s] = index
                    step += 1.0 / self.batch_size

                distribution['strata_ends'] = strata_ends

                res[partition_num] = distribution

            partition_num += 1

        return res

    def fix_index(self):
        """
        get next insert index
        :return: index, int
        """
        if self.record_size < self.capacity:#self.record_size <= self.size:
            self.record_size += 1
        if self.index % self.capacity == 0:
            self.isFull = True if len(self.buffer) == self.capacity else False
            if self.replace_flag:
                self.index = 1
                return self.index
            else:
                sys.stderr.write('Experience replay buff is full and replace is set to FALSE!\n')
                return -1
        else:
            self.index += 1
            return self.index

    def store(self, sample):
        """
        store experience, suggest that experience is a tuple of (s1, a, r, s2, t)
        so each experience is valid
        :param sample: maybe a tuple, or list
        :return: bool, indicate insert status
        """
        insert_index = self.fix_index()
        if insert_index > 0:
            if insert_index in self.buffer:
                del self.buffer[insert_index]
            self.buffer[insert_index] = sample
            # add to priority queue
            priority = self.priority_queue.get_max_priority()
            self.priority_queue.update(priority, insert_index)
            return True
        else:
            sys.stderr.write('Insert failed\n')
            return False

    def retrieve(self, indices):
        """
        get experience from indices
        :param indices: list of experience id
        :return: experience replay sample
        """
        return [self.buffer[v] for v in indices]

    def rebalance(self):
        """
        rebalance priority queue
        :return: None
        """
        self.priority_queue.balance_tree()

    def update_priorities(self, idxes, priorities):
        """
        update priority according indices and deltas
        :param idxes: list of experience id
        :param priorities: list of delta, order correspond to indices
        :return: None
        """
        assert len(idxes) == len(priorities)
        for i in range(0, len(idxes)):
            self.priority_queue.update(math.fabs(priorities[i]), idxes[i])

    def update_beta(self, idx):
        v = self.beta_start + idx * (1.0 - self.beta_start) / self.beta_frames
        self.beta = min(1.0, v)
        return self.beta

    def _add(self, sample):
        self.store(sample)

    def sample(self, _):
        """
        sample a mini batch from experience replay
        :return: experience, list, samples
        :return: w, list, weights
        :return: rank_e_id, list, samples id, used for update priority
        """
        if self.record_size < self.learn_start:
            sys.stderr.write('Record size less than learn start! Sample failed\n')
            return False, False, False
        dist_index = int(math.floor(1.0 * self.record_size / self.capacity * self.partition_num))
        partition_size = int(math.floor(1.0 * self.capacity / self.partition_num))
        partition_max = dist_index * partition_size
        distribution = self.distributions[dist_index]
        rank_list = []
        # sample from k segments
        for n in range(1, self.batch_size + 1):
            if distribution['strata_ends'][n] + 1 <= distribution['strata_ends'][n + 1]:
                index = random.randint(distribution['strata_ends'][n] + 1,
                                       distribution['strata_ends'][n + 1])
            else:
                index = random.randint(distribution['strata_ends'][n + 1],
                                       distribution['strata_ends'][n] + 1)
            rank_list.append(index)

        # beta, increase by global_step, max 1
        # beta = min(self.beta_zero + (global_step - self.learn_start - 1) * self.beta_grad, 1)
        beta = self.beta
        # find all alpha pow, notice that pdf is a list, start from 0
        alpha_pow = [distribution['pdf'][v - 1] for v in rank_list]
        # w = (N * P(i)) ^ (-beta) / max w
        w = np.power(np.array(alpha_pow) * partition_max, -beta)
        w_max = max(w)
        w = np.divide(w, w_max)
        # rank list is priority id
        # convert to experience id
        rank_e_id = self.priority_queue.priority_to_experience(rank_list)
        # get experience id according rank_e_id
        experience = self.retrieve(rank_e_id)
        return experience, rank_e_id, w


class BinaryHeap(object):
    def __init__(self, priority_size=100, priority_init=None, replace=True):
        self.e2p = {}
        self.p2e = {}
        self.replace = replace

        if priority_init is None:
            self.priority_queue = {}
            self.size = 0
            self.max_size = priority_size
        else:
            # not yet test
            self.priority_queue = priority_init
            self.size = len(self.priority_queue)
            self.max_size = None or self.size

            experience_list = list(map(lambda x: self.priority_queue[x], self.priority_queue))
            self.p2e = list_to_dict(experience_list)
            self.e2p = exchange_key_value(self.p2e)
            for i in range(int(self.size / 2), -1, -1):
                self.down_heap(i)

    def __repr__(self):
        """
        :return: string of the priority queue, with level info
        """
        if self.size == 0:
            return 'No element in heap!'
        to_string = ''
        level = -1
        max_level = int(math.floor(math.log(self.size, 2)))

        for i in range(1, self.size + 1):
            now_level = int(math.floor(math.log(i, 2)))
            if level != now_level:
                to_string = to_string + ('\n' if level != -1 else '') \
                            + '    ' * (max_level - now_level)
                level = now_level

            to_string = to_string + '%.2f ' % self.priority_queue[i][1] + '    ' * (max_level - now_level)

        return to_string

    def check_full(self):
        return self.size > self.max_size

    def _insert(self, priority, e_id):
        """
        insert new experience id with priority
        (maybe don't need get_max_priority and implement it in this function)
        :param priority: priority value
        :param e_id: experience id
        :return: bool
        """
        self.size += 1

        if self.check_full() and not self.replace:
            sys.stderr.write('Error: no space left to add experience id %d with priority value %f\n' % (e_id, priority))
            return False
        else:
            self.size = min(self.size, self.max_size)

        self.priority_queue[self.size] = (priority, e_id)
        self.p2e[self.size] = e_id
        self.e2p[e_id] = self.size

        self.up_heap(self.size)
        return True

    def update(self, priority, e_id):
        """
        update priority value according its experience id
        :param priority: new priority value
        :param e_id: experience id
        :return: bool
        """
        if e_id in self.e2p:
            p_id = self.e2p[e_id]
            self.priority_queue[p_id] = (priority, e_id)
            self.p2e[p_id] = e_id

            self.down_heap(p_id)
            self.up_heap(p_id)
            return True
        else:
            # this e id is new, do insert
            return self._insert(priority, e_id)

    def get_max_priority(self):
        """
        get max priority, if no experience, return 1
        :return: max priority if size > 0 else 1
        """
        if self.size > 0:
            return self.priority_queue[1][0]
        else:
            return 1

    def pop(self):
        """
        pop out the max priority value with its experience id
        :return: priority value & experience id
        """
        if self.size == 0:
            sys.stderr.write('Error: no value in heap, pop failed\n')
            return False, False

        pop_priority, pop_e_id = self.priority_queue[1]
        self.e2p[pop_e_id] = -1
        # replace first
        last_priority, last_e_id = self.priority_queue[self.size]
        self.priority_queue[1] = (last_priority, last_e_id)
        self.size -= 1
        self.e2p[last_e_id] = 1
        self.p2e[1] = last_e_id

        self.down_heap(1)

        return pop_priority, pop_e_id

    def up_heap(self, i):
        """
        upward balance
        :param i: tree node i
        :return: None
        """
        if i > 1:
            parent = int(math.floor(i / 2))
            if self.priority_queue[parent][0] < self.priority_queue[i][0]:
                tmp = self.priority_queue[i]
                self.priority_queue[i] = self.priority_queue[parent]
                self.priority_queue[parent] = tmp
                # change e2p & p2e
                self.e2p[self.priority_queue[i][1]] = i
                self.e2p[self.priority_queue[parent][1]] = parent
                self.p2e[i] = self.priority_queue[i][1]
                self.p2e[parent] = self.priority_queue[parent][1]
                # up heap parent
                self.up_heap(parent)

    def down_heap(self, i):
        """
        downward balance
        :param i: tree node i
        :return: None
        """
        if i < self.size:
            greatest = i
            left, right = i * 2, i * 2 + 1
            if left < self.size and self.priority_queue[left][0] > self.priority_queue[greatest][0]:
                greatest = left
            if right < self.size and self.priority_queue[right][0] > self.priority_queue[greatest][0]:
                greatest = right

            if greatest != i:
                tmp = self.priority_queue[i]
                self.priority_queue[i] = self.priority_queue[greatest]
                self.priority_queue[greatest] = tmp
                # change e2p & p2e
                self.e2p[self.priority_queue[i][1]] = i
                self.e2p[self.priority_queue[greatest][1]] = greatest
                self.p2e[i] = self.priority_queue[i][1]
                self.p2e[greatest] = self.priority_queue[greatest][1]
                # down heap greatest
                self.down_heap(greatest)

    def get_priority(self):
        """
        get all priority value
        :return: list of priority
        """
        return list(map(lambda x: x[0], self.priority_queue.values()))[0:self.size]

    def get_e_id(self):
        """
        get all experience id in priority queue
        :return: list of experience ids order by their priority
        """
        return list(map(lambda x: x[1], self.priority_queue.values()))[0:self.size]

    def balance_tree(self):
        """
        rebalance priority queue
        :return: None
        """
        sort_array = sorted(self.priority_queue.values(), key=lambda x: x[0], reverse=True)
        # reconstruct priority_queue
        self.priority_queue.clear()
        self.p2e.clear()
        self.e2p.clear()
        cnt = 1
        while cnt <= self.size:
            priority, e_id = sort_array[cnt - 1]
            self.priority_queue[cnt] = (priority, e_id)
            self.p2e[cnt] = e_id
            self.e2p[e_id] = cnt
            cnt += 1
        # sort the heap
        for i in range(int(math.floor(self.size / 2)), 1, -1):
            self.down_heap(i)

    def priority_to_experience(self, priority_ids):
        """
        retrieve experience ids by priority ids
        :param priority_ids: list of priority id
        :return: list of experience id
        """
        return [self.p2e[i] for i in priority_ids]


def list_to_dict(in_list):
    return dict((i, in_list[i]) for i in range(0, len(in_list)))


def exchange_key_value(in_dict):
    return dict((in_dict[i], i) for i in in_dict)