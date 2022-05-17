from collections import Counter
from itertools import islice
# import matplotlib.pyplot as plt
# import networkx as nx
import numpy as np
import datetime
import pickle
import random
import enum
import copy
import gym
import os
from gym import spaces
from typing import Optional

from a_configuration.a_base_config.a_environments.task_allocation.config_task_allocation import ConfigTaskAllocation


class CloudNetwork:
    def __init__(self, config):
        self.config = config

        # generate cloud servers
        self.servers = {}
        for server_id in range(self.config.NUM_CLOUD_SERVER):
            self.servers[server_id] = random.randint(
                    self.config.CLOUD_CPU_CAPACITY_MIN, self.config.CLOUD_CPU_CAPACITY_MAX)
        self.bandwidth = self.config.CLOUD_BANDWIDTH_CAPACITY

    def get_resource_remains(self):
        remaining_cpu_resource = sum([self.servers[server_id] for server_id in self.servers])

        return remaining_cpu_resource

    def __str__(self):
        remaining_cpu_resource = self.get_resource_remains()

        substrate_str = "[SUBST. CPU: {0:6.2f}%]".format(
            100 * remaining_cpu_resource / self.initial_total_cpu_capacity,
        )

        return substrate_str

    def __repr__(self):
        remaining_cpu_resource = self.get_resource_remains()

        substrate_str = "[SUBSTRATE CPU: {0:4}/{1:4}={2:6.2f}% ({3:2}~{4:3})".format(
                            remaining_cpu_resource, self.initial_total_cpu_capacity,
                            100 * remaining_cpu_resource / self.initial_total_cpu_capacity,
                            self.min_cpu_capacity, self.max_cpu_capacity
                        )

        return substrate_str


class EdgeNetwork:
    def __init__(self, config):
        self.config = config

        # generate cloud servers
        self.servers = {}
        for server_id in range(self.config.NUM_EDGE_SERVER):
            self.servers[server_id] = random.randint(
                self.config.EDGE_CPU_CAPACITY_MIN, self.config.EDGE_CPU_CAPACITY_MAX)
        self.bandwidth = self.config.EDGE_BANDWIDTH_CAPACITY

    def get_resource_remains(self):
        remaining_cpu_resource = sum([self.servers[server_id] for server_id in self.servers])

        return remaining_cpu_resource

    def __str__(self):
        remaining_cpu_resource = self.get_resource_remains()

        substrate_str = "[SUBST. CPU: {0:6.2f}%]".format(
            100 * remaining_cpu_resource / self.initial_total_cpu_capacity,
        )

        return substrate_str

    def __repr__(self):
        remaining_cpu_resource = self.get_resource_remains()

        substrate_str = "[SUBSTRATE CPU: {0:4}/{1:4}={2:6.2f}% ({3:2}~{4:3})".format(
            remaining_cpu_resource, self.initial_total_cpu_capacity,
            100 * remaining_cpu_resource / self.initial_total_cpu_capacity,
            self.min_cpu_capacity, self.max_cpu_capacity
        )

        return substrate_str


class Task:
    def __init__(self, config):
        self.config = config
        self.tasks = {}

        for task_id in range(self.config.NUM_TASK):
            data_size = int(random.randint(self.config.TASK_DATA_SIZE_MIN, self.config.TASK_DATA_SIZE_MAX))
            request_cpu = int(random.randint(self.config.TASK_CPU_REQUEST_MIN, self.config.TASK_CPU_REQUEST_MAX))
            request_latency = int(random.randint(self.config.TASK_LATENCY_REQUEST_MIN, self.config.TASK_LATENCY_REQUEST_MAX))
            self.tasks[task_id] = (data_size, request_cpu, request_latency)


class TaskAllocationEnvironment(gym.Env):
    def __init__(self, config):
        self.config = config
        self.task = Task(self.config)
        self.cloud_net = CloudNetwork(self.config)
        self.edge_net = EdgeNetwork(self.config)

        self.state = None
        self.task_id = None
        self.num_remain_task = None
        self.cloud_server_remain_cpu = None
        self.edge_server_remain_cpu = None
        self.cloud_server_cpu_list = None
        self.edge_server_cpu_list = None
        self.total_server_cpu_list = None
        self.cloud_bandwidth = None
        self.edge_bandwidth = None
        self.selected_num_cloud_server = None
        self.selected_num_edge_server = None
        self.selected_cloud_server_cpu = None
        self.selected_edge_server_cpu = None
        self.selected_cloud_bandwidth = None
        self.selected_edge_bandwidth = None

        self.revenue = None
        self.resource_util = None

        self.action_space = spaces.Discrete(self.config.NUM_CLOUD_SERVER + self.config.NUM_EDGE_SERVER)

        self.obs_space = self.config.NUM_TASK + self.config.NUM_EDGE_SERVER + self.config.NUM_CLOUD_SERVER + 6
        self.observation_space = spaces.Box(low=-1.0, high=10000.0, shape=((self.obs_space * 3,)))

    def reset(self, *, seed: Optional[int] = None, return_info: bool = False, options: Optional[dict] = None,):
        self.task = Task(self.config)
        self.cloud_net = CloudNetwork(self.config)
        self.edge_net = EdgeNetwork(self.config)
        self.cloud_bandwidth = self.cloud_net.bandwidth
        self.edge_bandwidth = self.edge_net.bandwidth

        self.selected_num_cloud_server = 0
        self.selected_num_edge_server = 0
        self.selected_cloud_server_cpu = 0
        self.selected_edge_server_cpu = 0
        self.selected_cloud_bandwidth = 0
        self.selected_edge_bandwidth = 0
        self.revenue = 0
        self.resource_util = 0
        self.num_remain_task = self.config.NUM_TASK

        # Set the first task ID
        self.task_id = 0

        # Get the cloud and edge servers' total remain cpu
        self.cloud_server_remain_cpu = self.cloud_net.get_resource_remains()
        self.edge_server_remain_cpu = self.edge_net.get_resource_remains()
        # Get the cloud and edge servers' each remain cpu
        self.cloud_server_cpu_list = [self.cloud_net.servers[server_id] for server_id in self.cloud_net.servers]
        self.edge_server_cpu_list = [self.edge_net.servers[server_id] for server_id in self.edge_net.servers]

        # Configurate initial state
        self.configurate_state()

        # Old version 1D state
        # [task_request_cpu, total_cloud_remain_cpu, total_edge_remain_cpu, each_cloud_remain_cpu, each_edge_remain_cpu]
        # self.state = [self.task.tasks[self.task_id][1], self.cloud_server_remain_cpu, self.edge_server_remain_cpu]
        self.total_server_cpu_list = self.cloud_server_cpu_list + self.edge_server_cpu_list
        # self.state += self.total_server_cpu_list

        observation = self.observation()

        info = dict()

        if return_info:
            return observation, info
        else:
            return observation

    def observation(self):
        observation = copy.deepcopy(self.state.flatten()) / (self.config.CLOUD_BANDWIDTH_CAPACITY + self.config.EDGE_BANDWIDTH_CAPACITY)

        return observation

    def step(self, action_idx):
        total_delay = 0
        request_bandwidth = 0
        reward = 0
        done = False
        info = {}

        # Calculate delay
        total_delay = self.calculate_delay(action_idx)
        # print("delay: ", total_delay)
        # print("task latency: ", self.task.tasks[self.task_id][2])

        # Calculate request bandwidth
        request_bandwidth = self.task.tasks[self.task_id][0] / self.task.tasks[self.task_id][2]
        # print("Request_bandwidth: ", request_bandwidth)

        # Reward
        if total_delay > self.task.tasks[self.task_id][2] or \
                self.total_server_cpu_list[action_idx] < self.task.tasks[self.task_id][1]:
            reward = (self.task.tasks[self.task_id][2] - total_delay) * self.resource_util
        else:
            self.num_remain_task -= 1
            # Apply action and update cloud and edge servers' cpu info
            self.total_server_cpu_list[action_idx] -= self.task.tasks[self.task_id][1]
            # select cloud
            if self.config.NUM_CLOUD_SERVER - action_idx > 0:
                self.cloud_server_remain_cpu -= self.task.tasks[self.task_id][1]
                self.cloud_bandwidth -= request_bandwidth
                self.cloud_server_cpu_list[action_idx] -= self.task.tasks[self.task_id][1]
                self.selected_cloud_bandwidth += request_bandwidth
                self.selected_cloud_server_cpu += self.task.tasks[self.task_id][1]
                self.selected_num_cloud_server += 1
            # select edge
            else:
                self.edge_server_remain_cpu -= self.task.tasks[self.task_id][1]
                self.edge_bandwidth -= request_bandwidth
                self.edge_server_cpu_list[action_idx - self.config.NUM_CLOUD_SERVER] -= self.task.tasks[self.task_id][1]
                self.selected_edge_bandwidth += request_bandwidth
                self.selected_edge_server_cpu += self.task.tasks[self.task_id][1]
                self.selected_num_edge_server += 1

            # Calculate resource utilization
            self.resource_util += self.task.tasks[self.task_id][1] + request_bandwidth
            reward = (self.task.tasks[self.task_id][2] - total_delay) * self.resource_util

        self.task_id += 1
        if self.config.NUM_TASK <= self.task_id:
            done = True
            # Generate next state
            self.task_id -= 1
            self.configurate_state()
        else:
            done = False
            # Generate next state
            self.configurate_state()

        observation = self.observation()

        return observation, reward, done, info

    def configurate_state(self):
        self.state = [[
            self.num_remain_task,
            self.cloud_server_remain_cpu + self.edge_server_remain_cpu,
            self.cloud_bandwidth + self.edge_bandwidth
        ]]
        self.state.append([self.config.NUM_CLOUD_SERVER, self.cloud_server_remain_cpu, self.cloud_bandwidth])
        for idx in range(self.config.NUM_CLOUD_SERVER):
            self.state.append([idx, self.cloud_server_cpu_list[idx], self.cloud_bandwidth])
        self.state.append([self.config.NUM_EDGE_SERVER, self.edge_server_remain_cpu, self.edge_bandwidth])
        for idx in range(self.config.NUM_EDGE_SERVER):
            self.state.append([idx, self.edge_server_cpu_list[idx], self.edge_bandwidth])
        self.state.append(
            [self.selected_num_cloud_server, self.selected_cloud_server_cpu, self.selected_cloud_bandwidth])
        self.state.append([self.selected_num_edge_server, self.selected_edge_server_cpu, self.selected_edge_bandwidth])
        self.state.append(
            [self.task.tasks[self.task_id][0], self.task.tasks[self.task_id][1], self.task.tasks[self.task_id][2]])
        for task_id in range(self.config.NUM_TASK):
            self.state.append([self.task.tasks[task_id][0], self.task.tasks[task_id][1], self.task.tasks[task_id][2]])
        self.state = np.asarray(self.state)

    def calculate_delay(self, action_idx):
        # 1. Transmission delay
        trans_delay = 0
        edge_trans_delay = (self.task.tasks[self.task_id][0] / self.edge_bandwidth) + \
                           (self.task.tasks[self.task_id][0] / self.edge_bandwidth)
        cloud_trans_delay = edge_trans_delay + \
                            (self.task.tasks[self.task_id][0] / self.cloud_bandwidth) + \
                            (self.task.tasks[self.task_id][0] / self.cloud_bandwidth)
        if self.config.NUM_CLOUD_SERVER - action_idx > 0:  # select cloud
            trans_delay = cloud_trans_delay
        else:  # select edge
            trans_delay = edge_trans_delay

        # 2. Propagation delay
        prop_delay = 0
        # select cloud
        if self.config.NUM_CLOUD_SERVER - action_idx > 0:
            prop_delay += 50
        # select edge
        else:
            prop_delay += 5

        # 3. Processing delay
        proc_delay = 0
        proc_delay = self.task.tasks[self.task_id][1] / self.total_server_cpu_list[action_idx]

        total_delay = trans_delay + prop_delay + proc_delay

        return total_delay

class Dummy_Agent:
    def __init__(self, config):
        self.config = config

    def get_action(self, state):
        assert state is not None
        available_action_ids = range(self.config.NUM_CLOUD_SERVER + self.config.NUM_EDGE_SERVER)
        action_id = random.choice(available_action_ids)
        return action_id

class Heuristic_Agent:
    def __init__(self, config):
        self.config = config

    def get_action(self, state):
        assert state is not None
        available_action_ids = range(self.config.NUM_CLOUD_SERVER + self.config.NUM_EDGE_SERVER)
        action_id = random.choice(available_action_ids)
        return action_id

def run_env():
    config = ConfigTaskAllocation()

    env = TaskAllocationEnvironment(config)
    agent = Dummy_Agent(config)
    # agent = Heuristic_Agent(config)
    state = env.reset()

    print("task", env.task.tasks)
    print("Cloud Net", env.cloud_net.servers)
    print("Edge Net", env.edge_net.servers)
    print()

    # print("reset state", state)
    # action_idx = agent.get_action(state)
    # state, reward, done, info = env.step(action_idx)
    # print("step 0, action: ", action_idx, state, reward, done)
    # state, reward, done, info = env.step(action_idx)
    # action_idx = agent.get_action(state)
    # print("step 1, action: ", action_idx, state, reward, done)
    # action_idx = agent.get_action(state)
    # state, reward, done, info = env.step(action_idx)
    # print("step 2, action: ", action_idx, state, reward, done)

    print("reset state", state)
    print(type(state), state.shape)
    action_idx = agent.get_action(state)
    state, reward, done, info = env.step(action_idx)
    print("step 0, action: ", action_idx, reward, done)
    state, reward, done, info = env.step(action_idx)
    action_idx = agent.get_action(state)
    print("step 1, action: ", action_idx, reward, done)
    action_idx = agent.get_action(state)
    state, reward, done, info = env.step(action_idx)
    print("step 2, action: ", action_idx, reward, done)


if __name__ == "__main__":
    run_env()