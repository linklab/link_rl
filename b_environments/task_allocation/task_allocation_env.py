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
        self.initial_s_cpu_capacity = {}
        self.initial_s_bw_capacity = {}
        self.initial_s_node_total_bandwidth = {}

        self.initial_total_cpu_capacity = 0
        self.initial_total_bandwidth_capacity = 0

        # corresponding CPU and BANDWIDTH resources of it are real numbers uniformly distributed from 50 to 100
        self.min_cpu_capacity = 1.0e10
        self.max_cpu_capacity = 0.0
        self.min_bandwidth_capacity = 1.0e10
        self.max_bandwidth_capacity = 0.0

        # generate cloud servers
        self.servers = {}
        for server_id in range(self.config.NUM_CLOUD_SERVER):
            self.servers[server_id] = random.randint(
                    self.config.CLOUD_CPU_CAPACITY_MIN, self.config.CLOUD_CPU_CAPACITY_MAX)

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
        self.initial_s_cpu_capacity = {}
        self.initial_s_bw_capacity = {}
        self.initial_s_node_total_bandwidth = {}

        self.initial_total_cpu_capacity = 0
        self.initial_total_bandwidth_capacity = 0

        # corresponding CPU and BANDWIDTH resources of it are real numbers uniformly distributed from 50 to 100
        self.min_cpu_capacity = 1.0e10
        self.max_cpu_capacity = 0.0
        self.min_bandwidth_capacity = 1.0e10
        self.max_bandwidth_capacity = 0.0

        # generate cloud servers
        self.servers = {}
        for server_id in range(self.config.NUM_EDGE_SERVER):
            self.servers[server_id] = random.randint(
                self.config.EDGE_CPU_CAPACITY_MIN, self.config.EDGE_CPU_CAPACITY_MAX)

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
        self.revenue_cpu = 0
        self.tasks = {}

        for task_id in range(self.config.NUM_TASK):
            data_size = int(random.randint(self.config.TASK_DATA_SIZE_MIN, self.config.TASK_DATA_SIZE_MAX))
            request_cpu = int(random.randint(self.config.TASK_CPU_REQUEST_MIN, self.config.TASK_CPU_REQUEST_MAX))
            request_latency = int(random.randint(self.config.TASK_LATENCY_REQUEST_MIN, self.config.TASK_LATENCY_REQUEST_MAX))
            self.tasks[task_id] = (data_size, request_cpu, request_latency)

        for task_id in range(self.config.NUM_TASK):
            for _, request_cpu, _ in self.tasks[task_id]:
                self.revenue += request_cpu
        self.cost = None


# class TaskAllocationEnvironment:
#     def __init__(self, config):
#         self.config = config
#
#
#     def reset(self):  # 에피소드 시작할 때 한번 호출
#
#
#     def step(self, v_actions, actions, assigned_vnr_id_list):



def run_env():
    config = ConfigTaskAllocation()
    task = Task(config)
    cloudNet = CloudNetwork(config)
    edgeNet = EdgeNetwork(config)

if __name__ == "__main__":
    run_env()