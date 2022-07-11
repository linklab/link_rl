import random
from typing import Optional

import gym
from gym import spaces
import enum
import copy
import numpy as np

from link_rl.a_configuration.a_base_config.a_environments.task_allocation.config_basic_task_allocation import ConfigBasicTaskAllocation0
from link_rl.h_utils.stats import CustomEnvStat


class DoneReasonType0(enum.Enum):
    TYPE_1 = "Resource Limit Exceeded"
    TYPE_2 = "Resource Remains"
    TYPE_3 = "All Tasks Selected"


class EnvironmentBasicTaskScheduling0Stat(CustomEnvStat):
    def __init__(self):
        super(EnvironmentBasicTaskScheduling0Stat, self).__init__()
        self.test_utilization_lst = []

        self.test_last_avg_utilization = 0.0

        self.train_last_utilization = 0.0

    def test_reset(self):
        self.test_utilization_lst.clear()

    def test_episode_done(self, info):
        self.test_utilization_lst.append(info["Utilization"])

    def test_evaluate(self):
        self.test_last_avg_utilization = np.average(self.test_utilization_lst)

    def test_evaluation_str(self):
        return "Utilization: {0:.2f}".format(self.test_last_avg_utilization)

    def train_evaluate(self, last_train_env_info):
        self.train_last_utilization = last_train_env_info["Utilization"]

    def train_evaluation_str(self):
        _train_evaluation_str = "Resource Utilization: {0:>4.2f}".format(self.train_last_utilization)
        return _train_evaluation_str

    def add_wandb_log(self, log_dict):
        log_dict["Utilization"] = self.train_last_utilization
        log_dict["[TEST] Utilization"] = self.test_last_avg_utilization


class EnvironmentBasicTaskScheduling0(gym.Env):
    def __init__(self, config):
        self.NUM_TASKS = config.NUM_TASK
        self.NUM_RESOURCES = config.NUM_RESOURCES
        self.INITIAL_RESOURCES_CAPACITY = config.INITIAL_RESOURCES_CAPACITY
        self.SUM_RESOURCE_CAPACITY = sum(self.INITIAL_RESOURCES_CAPACITY)
        self.LOW_DEMAND_RESOURCE_AT_TASK = config.LOW_DEMAND_RESOURCE_AT_TASK
        self.HIGH_DEMAND_RESOURCE_AT_TASK = config.HIGH_DEMAND_RESOURCE_AT_TASK
        self.INITIAL_TASK_DISTRIBUTION_FIXED = config.INITIAL_TASK_DISTRIBUTION_FIXED

        self.internal_state = None
        self.tasks_selected = None
        self.resource_of_all_tasks_selected = None
        self.min_task_demand = None
        self.num_step = None
        self.actions_sequence = None
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(
            low=-1.0, high=1000.0,
            shape=((self.NUM_TASKS + 1) * (self.NUM_RESOURCES + 2),)
        )

        if self.INITIAL_TASK_DISTRIBUTION_FIXED:
            self.fixed_initial_internal_state = self.get_initial_internal_state()

    # Last Row in State
    # 0: Always 0
    # 1: initially set to LIMIT_WEIGHT_KNAPSACK, and decrease by an item's weight whenever the item is selected
    # 2: initially set to 0, and increase by an item's weight whenever the item is selected
    # 3: initially set to 0, and increase by an item's value whenever the item is selected
    def get_initial_internal_state(self):
        state = np.zeros(shape=(self.NUM_TASKS + 1, self.NUM_RESOURCES + 2), dtype=int)

        for task_idx in range(self.NUM_TASKS):
            resource_demand = np.random.randint(
                low=self.LOW_DEMAND_RESOURCE_AT_TASK,
                high=self.HIGH_DEMAND_RESOURCE_AT_TASK,
                size=(1, self.NUM_RESOURCES)
            )
            state[task_idx][2:] = resource_demand

        self.min_task_demand = []

        for i in range(self.NUM_RESOURCES):
            self.min_task_demand.append(state[:-1, 2 + i:].min())

        state[-1][1] = sum(self.INITIAL_RESOURCES_CAPACITY)
        state[-1][2:] = np.array(self.INITIAL_RESOURCES_CAPACITY)

        return state

    def observation(self):
        observation = copy.deepcopy(self.internal_state.flatten()) / self.SUM_RESOURCE_CAPACITY
        return observation

    def reset(self, *, seed: Optional[int] = None, return_info: bool = False, options: Optional[dict] = None,):
        if self.INITIAL_TASK_DISTRIBUTION_FIXED:
            assert self.fixed_initial_internal_state is not None
            self.internal_state = copy.deepcopy(self.fixed_initial_internal_state)
        else:
            self.internal_state = self.get_initial_internal_state()

        self.tasks_selected = []
        self.resource_of_all_tasks_selected = [0 for _ in range(self.NUM_RESOURCES)]
        self.actions_sequence = []

        self.num_step = 0
        self.internal_state[self.num_step][0] = 1
        observation = self.observation()

        info = dict()
        info['internal_state'] = copy.deepcopy(self.internal_state)

        if return_info:
            return observation, info
        else:
            return observation

    def check_future_allocation_possible(self):
        possible = False

        for task in self.internal_state[self.num_step + 1: self.NUM_TASKS]:
            resources = task[2:]

            for i in range(self.NUM_RESOURCES):
                if resources[i] <= self.internal_state[-1][2 + i]:
                    possible = True
                    break

        return possible

    def step(self, action_idx):
        self.actions_sequence.append(action_idx)
        info = dict()
        step_resource = self.internal_state[self.num_step][2:]

        if action_idx == 1:
            self.tasks_selected.append(self.num_step)
            for i in range(self.NUM_RESOURCES):
                self.resource_of_all_tasks_selected[i] += step_resource[i]
                self.internal_state[-1][2 + i] -= step_resource[i]
                self.internal_state[-1][1] -= step_resource[i]

            self.internal_state[self.num_step][1] = 1
            self.internal_state[self.num_step][2:] = -1

        possible = self.check_future_allocation_possible()

        done = False
        if self.num_step == self.NUM_TASKS - 1 or not possible:
            done = True

            if 0 not in self.internal_state[:, 1]:
                info['DoneReasonType'] = DoneReasonType0.TYPE_3     # All Tasks Selected
            else:
                over_resource = False
                for i in range(self.NUM_RESOURCES):
                    if self.resource_of_all_tasks_selected[i] > self.INITIAL_RESOURCES_CAPACITY[i]:
                        info['DoneReasonType'] = DoneReasonType0.TYPE_1         # Resource Limit Exceeded
                        over_resource = True
                        break

                if not over_resource:
                    info['DoneReasonType'] = DoneReasonType0.TYPE_2     # Resource Remains

            self.internal_state[self.num_step][0] = 0
            self.internal_state[-1][0] = 1
        else:
            self.internal_state[self.num_step][0] = 0
            self.num_step += 1
            self.internal_state[self.num_step][0] = 1

        observation = self.observation()

        info['Actions sequence'] = self.actions_sequence
        info['Tasks selected'] = self.tasks_selected
        info['Resources allocated'] = self.resource_of_all_tasks_selected
        info['Limit'] = self.INITIAL_RESOURCES_CAPACITY
        info['Utilization'] = sum(self.resource_of_all_tasks_selected) / self.SUM_RESOURCE_CAPACITY if sum(self.resource_of_all_tasks_selected) / self.SUM_RESOURCE_CAPACITY <= 1.0 else 0.0
        info['internal_state'] = copy.deepcopy(self.internal_state)

        if done:
            reward = self.reward(done_type=info['DoneReasonType'])
        else:
            reward = self.reward(done_type=None)

        return observation, reward, done, info

    def reward(self, done_type=None):
        if done_type is None:  # Normal Step
            resource_efficiency_reward = 0.0
            mission_complete_reward = 0.0
            misbehavior_reward = 0.0

        elif done_type == DoneReasonType0.TYPE_1:  # Resource Limit Exceeded
            resource_efficiency_reward = 0.0
            mission_complete_reward = 0.0
            misbehavior_reward = -1.0

        elif done_type == DoneReasonType0.TYPE_2:  # Resource Remains
            resource_efficiency_reward = sum(self.resource_of_all_tasks_selected) / self.SUM_RESOURCE_CAPACITY
            mission_complete_reward = 0.0
            misbehavior_reward = 0.0

        elif done_type == DoneReasonType0.TYPE_3:  # All Tasks Selected
            resource_efficiency_reward = sum(self.resource_of_all_tasks_selected) / self.SUM_RESOURCE_CAPACITY
            mission_complete_reward = 1.0
            misbehavior_reward = 0.0

        else:
            raise ValueError()

        return resource_efficiency_reward + mission_complete_reward + misbehavior_reward

    def print_internal_state(self):
        print("----------------------state-----------------------------")
        print(self.internal_state)
        print("--------------------------------------------------------")


def run_env():
    class Dummy_Agent:
        def get_action(self, observation):
            assert observation is not None
            available_action_ids = [0, 1]
            action_id = random.choice(available_action_ids)
            return action_id

    print("START RUN!!!")
    agent = Dummy_Agent()

    config = ConfigBasicTaskAllocation0()
    # config.NUM_TASK = 3
    # config.NUM_RESOURCES = 3
    # config.INITIAL_RESOURCES_CAPACITY = [100, 200, 300]
    # config.LOW_DEMAND_RESOURCE_AT_TASK = [10, 20, 30]
    # config.HIGH_DEMAND_RESOURCE_AT_TASK = [20, 30, 40]
    config.NUM_TASK = 3
    config.NUM_RESOURCES = 2
    config.INITIAL_RESOURCES_CAPACITY = [config.NUM_TASK * 10 for _ in range(config.NUM_RESOURCES)]
    config.LOW_DEMAND_RESOURCE_AT_TASK = [10]
    config.HIGH_DEMAND_RESOURCE_AT_TASK = [20]

    env = EnvironmentBasicTaskScheduling0(config)

    for i in range(2):
        observation, info = env.reset(return_info=True)
        done = False
        print("EPISODE: {0} ".format(i + 1) + "#" * 50)
        while not done:
            action = agent.get_action(observation)
            next_observation, reward, done, next_info = env.step(action)

            print("Observation: \n{0}, \nAction: {1}, next_observation: \n{2}, Reward: {3}, Done: {4} ".format(
                info['internal_state'], action, next_info['internal_state'], reward, done
            ), end="")
            if done:
                print("({0}: {1})\n".format(next_info['DoneReasonType'], next_info['DoneReasonType'].value))
            else:
                print("\n")
            observation = next_observation
            info = next_info


if __name__ == "__main__":
    run_env()