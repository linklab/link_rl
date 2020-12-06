import argparse
import os
import time
import operator
from datetime import timedelta
import numpy as np
import collections

import torch
import torch.nn as nn

from common.fast_rl.common.statistics import StatisticsForValueBasedRL, StatisticsForPolicyBasedRL


class SMAQueue:
    """
    Queue of fixed size with mean, max, min operations
    """
    def __init__(self, size):
        self.queue = collections.deque()
        self.size = size

    def __iadd__(self, other):
        if isinstance(other, (list, tuple)):
            self.queue.extend(other)
        else:
            self.queue.append(other)
        while len(self.queue) > self.size:
            self.queue.popleft()
        return self

    def __len__(self):
        return len(self.queue)

    def __repr__(self):
        return "SMAQueue(size=%d)" % self.size

    def __str__(self):
        return "SMAQueue(size=%d, len=%d)" % (self.size, len(self.queue))

    def min(self):
        if not self.queue:
            return None
        return np.min(self.queue)

    def mean(self):
        if not self.queue:
            return None
        return np.mean(self.queue)

    def max(self):
        if not self.queue:
            return None
        return np.max(self.queue)


class SpeedMonitor:
    def __init__(self, batch_size, autostart=True):
        self.batch_size = batch_size
        self.start_ts = None
        self.batches = None
        if autostart:
            self.reset()

    def epoch(self):
        if self.epoches is not None:
            self.epoches += 1

    def batch(self):
        if self.batches is not None:
            self.batches += 1

    def reset(self):
        self.start_ts = time.time()
        self.batches = 0
        self.epoches = 0

    def seconds(self):
        """
        Seconds since last reset
        :return:
        """
        return time.time() - self.start_ts

    def samples_per_sec(self):
        """
        Calculate samples per second since last reset() call
        :return: float count samples per second or None if not started
        """
        if self.start_ts is None:
            return None
        secs = self.seconds()
        if abs(secs) < 1e-5:
            return 0.0
        return (self.batches + 1) * self.batch_size / secs

    def epoch_time(self):
        """
        Calculate average epoch time
        :return: timedelta object
        """
        if self.start_ts is None:
            return None
        s = self.seconds()
        if self.epoches > 0:
            s /= self.epoches + 1
        return timedelta(seconds=s)

    def batch_time(self):
        """
        Calculate average batch time
        :return: timedelta object
        """
        if self.start_ts is None:
            return None
        s = self.seconds()
        if self.batches > 0:
            s /= self.batches + 1
        return timedelta(seconds=s)


class WeightedMSELoss(nn.Module):
    def __init__(self, size_average=True):
        super(WeightedMSELoss, self).__init__()
        self.size_average = size_average

    def forward(self, input, target, weights=None):
        if weights is None:
            return nn.MSELoss(self.size_average)(input, target)

        loss_rows = (input - target) ** 2
        if len(loss_rows.size()) != 1:
            loss_rows = torch.sum(loss_rows, dim=1)
        res = (weights * loss_rows).sum()
        if self.size_average:
            res /= len(weights)
        return res


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


class TBMeanTracker:
    """
    TensorBoard value tracker: allows to batch fixed amount of historical values and write their mean into TB

    Designed and tested with pytorch-tensorboard in mind
    """
    def __init__(self, writer, batch_size):
        """
        :param writer: writer with close() and add_scalar() methods
        :param batch_size: integer size of batch to track
        """
        assert isinstance(batch_size, int)
        assert writer is not None
        self.writer = writer
        self.batch_size = batch_size

    def __enter__(self):
        self._batches = collections.defaultdict(list)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.writer.close()

    @staticmethod
    def _as_float(value):
        assert isinstance(value, (float, int, np.ndarray, np.generic, torch.autograd.Variable)) or torch.is_tensor(value)
        tensor_val = None
        if isinstance(value, torch.autograd.Variable):
            tensor_val = value.data
        elif torch.is_tensor(value):
            tensor_val = value

        if tensor_val is not None:
            return tensor_val.float().mean().item()
        elif isinstance(value, np.ndarray):
            return float(np.mean(value))
        else:
            return float(value)

    def track(self, param_name, value, iter_index):
        assert isinstance(param_name, str)
        assert isinstance(iter_index, int)

        data = self._batches[param_name]
        data.append(self._as_float(value))

        if len(data) >= self.batch_size:
            self.writer.add_scalar(param_name, np.mean(data), iter_index)
            data.clear()


class RewardTracker:
    def __init__(self, params, frame=True, stat=None, worker_id=None):
        self.params = params
        self.min_ts_diff = 1    # 1 second
        self.stop_mean_episode_reward = params.STOP_MEAN_EPISODE_REWARD
        self.stat = stat
        self.average_size_for_stats = params.AVG_EPISODE_SIZE_FOR_STAT
        self.draw_viz = params.DRAW_VIZ
        self.frame = frame
        self.episode_reward_list = None
        self.done_episodes = 0
        self.mean_episode_reward = 0.0
        self.count_stop_condition_episode = 0
        self.worker_id = worker_id

    def __enter__(self):
        self.start_ts = time.time()
        self.ts = time.time()
        self.ts_frame = 0
        self.episode_reward_list = []
        return self

    def start_reward_track(self):
        self.__enter__()

    def __exit__(self, *args):
        pass

    def set_episode_reward(self, episode_reward, episode_done_step, epsilon, action_count=None, continuous_action_mean=None):
        assert not (action_count and continuous_action_mean)
        self.episode_reward_list.append(episode_reward)
        self.mean_episode_reward = np.mean(self.episode_reward_list[-self.average_size_for_stats:])

        current_ts = time.time()
        elapsed_time = current_ts - self.start_ts
        ts_diff = current_ts - self.ts

        is_print_performance = False

        if ts_diff > self.min_ts_diff:
            is_print_performance = True
            self.print_performance(
                episode_done_step, self.done_episodes, current_ts, ts_diff, self.mean_episode_reward, epsilon, elapsed_time, action_count
            )

        if self.mean_episode_reward > self.stop_mean_episode_reward:
            self.count_stop_condition_episode += 1
        else:
            self.count_stop_condition_episode = 0

        if self.count_stop_condition_episode >= self.params.STOP_CONDITION_CONTINUOUS_EPISODE:
            if not is_print_performance:
                self.print_performance(
                    episode_done_step, self.done_episodes, current_ts, ts_diff, self.mean_episode_reward, epsilon, elapsed_time, action_count
                )
            if self.frame:
                print("Solved in {0} frames and {1} episodes!".format(episode_done_step, self.done_episodes))
            else:
                print("Solved in {0} steps and {1} episodes!".format(episode_done_step, self.done_episodes))
            return True, self.mean_episode_reward

        self.done_episodes += 1

        return False, self.mean_episode_reward

    def print_performance(self, episode_done_step, done_episodes, current_ts, ts_diff, mean_episode_reward, epsilon, elapsed_time, action_count):
        speed = (episode_done_step - self.ts_frame) / ts_diff
        self.ts_frame = episode_done_step
        self.ts = current_ts

        if self.worker_id is not None:
            prefix = "[Worker ID: {0}]".format(self.worker_id)
        else:
            prefix = ""

        if isinstance(epsilon, tuple) or isinstance(epsilon, list):
            epsilon_str = "{0:5.3f}, {1:5.3f}".format(
                epsilon[0] if epsilon[0] else 0.0,
                epsilon[1] if epsilon[1] else 0.0
            )
        else:
            epsilon_str = "{0:5.3f}".format(
                epsilon if epsilon else 0.0,
            )

        if self.mean_episode_reward > self.stop_mean_episode_reward:
            episode_reward_str = "mean_{0}_episode_reward: {1:8.3f} (***{2}***)".format(
                self.average_size_for_stats,
                mean_episode_reward,
                self.count_stop_condition_episode + 1
            )
        else:
            episode_reward_str = "mean_{0}_episode_reward: {1:8.3f}".format(
                self.average_size_for_stats,
                mean_episode_reward
            )

        print(
            "{0}[{1:6}/{2}] Episode {3} done, episode_reward: {4:8.3f}, {5}, "
            "epsilon: {6}, speed: {7:7.2f} {8}, elapsed time: {9}".format(
                prefix,
                episode_done_step,
                self.params.MAX_GLOBAL_STEPS,
                done_episodes,
                self.episode_reward_list[-1],
                episode_reward_str,
                epsilon_str,
                speed,
                "fps" if self.frame else "steps/sec.",
                time.strftime("%Hh %Mm %Ss", time.gmtime(elapsed_time)),
        ), end="")

        if action_count:
            print(", {0}".format(action_count), flush=True)
        else:
            print("", flush=True)

        if self.draw_viz and self.stat:
            if isinstance(self.stat, StatisticsForValueBasedRL):
                self.stat.draw_performance(episode_done_step, mean_episode_reward, speed, epsilon)
            elif isinstance(self.stat, StatisticsForPolicyBasedRL):
                self.stat.draw_performance(episode_done_step, mean_episode_reward, speed)
            else:
                raise ValueError()


def distribution_projection(distribution, rewards, dones, v_min, v_max, n_atoms, gamma, device="cpu"):
    """
    Perform distribution projection aka Catergorical Algorithm from the
    "A Distributional Perspective on RL" paper
    """
    if torch.is_tensor(distribution):
        distribution = distribution.data.cpu().numpy()

    if torch.is_tensor(rewards):
        rewards = rewards.data.cpu().numpy()

    if torch.is_tensor(dones):
        dones = dones.cpu().numpy().astype(np.bool)

    batch_size = len(rewards)

    # to keep the result of the projection
    projected_distribution = np.zeros((batch_size, n_atoms), dtype=np.float32)

    # the width of every atom in our value range
    # v_max: 10, v_min: -10, n_atoms: 51 --> delta_z: 0.4
    delta_z = (v_max - v_min) / (n_atoms - 1)

    for atom in range(n_atoms):
        # reward: 1, v_min: -10, atom: 0, gamma: 0.99 --> v = 1 + (-10) * 0.99 = -8.9
        v = rewards + (v_min + atom * delta_z) * gamma
        tz_j = np.minimum(v_max, np.maximum(v_min, v))

        # tz_j: -8.9, v_min: -10, delta_z: 0.4 --> b_j = 2.75
        b_j = (tz_j - v_min) / delta_z
        l = np.floor(b_j).astype(np.int64)  # b_j: 2.75 --> l = 2
        u = np.ceil(b_j).astype(np.int64)   # b_j: 2.75 --> u = 3
        eq_mask = u == l
        projected_distribution[eq_mask, l[eq_mask]] += distribution[eq_mask, atom]

        ne_mask = u != l
        projected_distribution[ne_mask, l[ne_mask]] += distribution[ne_mask, atom] * (u - b_j)[ne_mask]
        projected_distribution[ne_mask, u[ne_mask]] += distribution[ne_mask, atom] * (b_j - l)[ne_mask]

    if dones.any():
        projected_distribution[dones] = 0.0
        tz_j = np.minimum(v_max, np.maximum(v_min, rewards[dones]))
        b_j = (tz_j - v_min) / delta_z
        l = np.floor(b_j).astype(np.int64)
        u = np.ceil(b_j).astype(np.int64)
        eq_mask = u == l
        eq_dones = dones.copy()
        eq_dones[dones] = eq_mask
        if eq_dones.any():
            projected_distribution[eq_dones, l[eq_mask]] = 1.0
        ne_mask = u != l
        ne_dones = dones.copy()
        ne_dones[dones] = ne_mask
        if ne_dones.any():
            projected_distribution[ne_dones, l[ne_mask]] = (u - b_j)[ne_mask]
            projected_distribution[ne_dones, u[ne_mask]] = (b_j - l)[ne_mask]

    return torch.FloatTensor(projected_distribution).to(device)
