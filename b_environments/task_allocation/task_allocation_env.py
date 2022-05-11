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


class TaskAllocationEnvironment:
    def __init__(self, config):
        self.config = config
        self.task = Task(self.config)
        self.cloud_net = CloudNetwork(self.config)
        self.edge_net = EdgeNetwork(self.config)

        self.state = None
        self.task_id = None
        self.cloud_server_remain_cpu = None
        self.edge_server_remain_cpu = None
        self.cloud_server_cpu_list = None
        self.edge_server_cpu_list = None
        self.total_server_cpu_list = None

        self.revenue = None
        self.resource_util = None

    def reset(self):  # ???????? ?????? ?? ???? ????
        self.task = Task(self.config)
        self.cloud_net = CloudNetwork(self.config)
        self.edge_net = EdgeNetwork(self.config)
        self.revenue = 0
        self.resource_util = 0

        # Set the first task ID
        self.task_id = 0

        # Get the cloud and edge servers' total remain cpu
        self.cloud_server_remain_cpu = self.cloud_net.get_resource_remains()
        self.edge_server_remain_cpu = self.edge_net.get_resource_remains()
        # Get the cloud and edge servers' each remain cpu
        self.cloud_server_cpu_list = [self.cloud_net.servers[server_id] for server_id in self.cloud_net.servers]
        self.edge_server_cpu_list = [self.edge_net.servers[server_id] for server_id in self.edge_net.servers]

        # Configurate initial state
        # [task_request_cpu, total_cloud_remain_cpu, total_edge_remain_cpu, each_cloud_remain_cpu, each_edge_remain_cpu]
        self.state = [self.task.tasks[self.task_id][1], self.cloud_server_remain_cpu, self.edge_server_remain_cpu]
        self.total_server_cpu_list = self.cloud_server_cpu_list + self.edge_server_cpu_list
        self.state += self.total_server_cpu_list

        return self.state

    def step(self, action_idx):
        total_delay = 0
        request_bandwidth = 0
        reward = 0
        done = False

        # Calculate delay
        total_delay = self.calculate_delay(action_idx)

        # Calculate request bandwidth
        request_bandwidth = self.task.tasks[self.task_id][0] / self.task.tasks[self.task_id][2]

        # Reward
        if total_delay > self.task.tasks[self.task_id][2] or \
                self.total_server_cpu_list[action_idx] < self.task.tasks[self.task_id][1]:
            reward = (self.task.tasks[self.task_id][2] - total_delay) * self.resource_util
        else:
            # Apply action and update cloud and edge servers' cpu info
            self.total_server_cpu_list[action_idx] -= self.task.tasks[self.task_id][1]
            # select cloud
            if self.config.NUM_CLOUD_SERVER - action_idx > 0:
                self.cloud_server_remain_cpu -= self.task.tasks[self.task_id][1]
            # select edge
            else:
                self.edge_server_remain_cpu -= self.task.tasks[self.task_id][1]
            # Calculate resource utilization
            self.resource_util += self.task.tasks[self.task_id][1] + request_bandwidth
            reward = (self.task.tasks[self.task_id][2] - total_delay) * self.resource_util

        self.task_id += 1
        if self.config.NUM_TASK <= self.task_id:
            done = True
        else:
            done = False
            # Generate next state
            self.state = [self.task.tasks[self.task_id][1], self.cloud_server_remain_cpu, self.edge_server_remain_cpu]
            self.state += self.total_server_cpu_list

        return self.state, reward, done

    def calculate_delay(self, action_idx):
        # 1. Transmission delay
        trans_delay = 0
        edge_trans_delay = (self.task.tasks[self.task_id][0] / self.edge_net.bandwidth) + \
                           (self.task.tasks[self.task_id][0] / self.edge_net.bandwidth)
        cloud_trans_delay = edge_trans_delay + \
                            (self.task.tasks[self.task_id][0] / self.cloud_net.bandwidth) + \
                            (self.task.tasks[self.task_id][0] / self.cloud_net.bandwidth)
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


def run_env():
    config = ConfigTaskAllocation()
    task = Task(config)
    cloud_net = CloudNetwork(config)
    edge_net = EdgeNetwork(config)
    env = TaskAllocationEnvironment(config)
    state = env.reset()

    print("check task", env.task.tasks)
    print("reset()", state)

    state, reward, done = env.step(0)
    print("step()", state, reward, done)
    state, reward, done = env.step(1)
    print("step()", state, reward, done)
    state, reward, done = env.step(2)
    print("step()", state, reward, done)


    # print("Initiate Task")
    # print(task.tasks)
    # print("Initiate CloudNet")
    # print(cloud_net.servers)
    # print(cloud_net.get_resource_remains())
    # print("Initiate EdgeNet")
    # print(edge_net.servers)
    # print(edge_net.get_resource_remains())


if __name__ == "__main__":
    run_env()