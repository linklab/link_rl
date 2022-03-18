from typing import Optional

import gym
from gym import spaces
import enum
import copy
import numpy as np

from a_configuration.a_base_config.a_environments.config_task_allocation import ConfigTakAllocation0

config = ConfigTakAllocation0()

NUM_TASKS = config.NUM_TASK
NUM_RESOURCES = config.NUM_RESOURCES
INITIAL_RESOURCES_CAPACITY = config.INITIAL_RESOURCES_CAPACITY
SUM_RESOURCE_CAPACITY = sum(INITIAL_RESOURCES_CAPACITY)
LOW_DEMAND_RESOURCE_AT_TASK = config.LOW_DEMAND_RESOURCE_AT_TASK
HIGH_DEMAND_RESOURCE_AT_TASK = config.HIGH_DEMAND_RESOURCE_AT_TASK
INITIAL_TASK_DISTRIBUTION_FIXED = config.INITIAL_TASK_DISTRIBUTION_FIXED
MAX_STEP = NUM_TASKS


class DoneReasonType0(enum.Enum):
    TYPE_1 = "Resource Limit Exceeded"
    TYPE_2 = "Not Enough Resource [NORMAL]"
    TYPE_3 = "All Tasks Selected"


class EnvironmentTaskScheduling0(gym.Env):
    def __init__(self):
        self.internal_state = None
        self.tasks_selected = None
        self.resource_of_all_tasks_selected = None
        self.min_task_demand = None
        self.num_step = None
        self.actions_sequence = None
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low=-1.0, high=1000, shape=((NUM_TASKS + 1)*(NUM_RESOURCES + 2),))

        if INITIAL_TASK_DISTRIBUTION_FIXED:
            self.fixed_initial_internal_state = self.get_initial_internal_state()

    def get_initial_internal_state(self):
        state = np.zeros(shape=(NUM_TASKS + 1, NUM_RESOURCES + 2), dtype=int)

        for task_idx in range(NUM_TASKS):
            resource_demand = np.random.randint(
                low=LOW_DEMAND_RESOURCE_AT_TASK, high=HIGH_DEMAND_RESOURCE_AT_TASK, size=(1, 1)
            )
            state[task_idx][2:] = resource_demand

        self.min_task_demand = state[:-1, 2].min()

        state[-1][2:] = np.array(INITIAL_RESOURCES_CAPACITY)

        return state

    def observation(self):
        observation = copy.deepcopy(self.internal_state.flatten()) / SUM_RESOURCE_CAPACITY
        return observation

    def reset(self, *, seed: Optional[int] = None, return_info: bool = False, options: Optional[dict] = None,):
        if INITIAL_TASK_DISTRIBUTION_FIXED:
            assert self.fixed_initial_internal_state is not None
            self.internal_state = copy.deepcopy(self.fixed_initial_internal_state)
        else:
            self.internal_state = self.get_initial_internal_state()

        self.tasks_selected = []
        self.resource_of_all_tasks_selected = 0
        self.num_step = 0

        self.actions_sequence = []

        observation = self.observation()
        observation[0] = 1

        info = dict()
        info['internal_state'] = self.internal_state

        if return_info:
            return observation, info
        else:
            return observation

    def step(self, action_idx):
        task_idx = self.num_step
        info = dict()
        self.num_step += 1
        self.internal_state[task_idx][0] = 0
        self.internal_state[task_idx+1][0] = 1

        step_resource = self.internal_state[task_idx][2]

        if action_idx == 1:
            self.actions_sequence.append(task_idx)
            self.tasks_selected.append(task_idx)

            self.resource_of_all_tasks_selected += step_resource

            self.internal_state[task_idx][1] = 1
            self.internal_state[-1][2] -= step_resource
            self.internal_state[-1][1] = self.resource_of_all_tasks_selected

        possible = False

        for _, _, resource in self.internal_state[self.num_step:NUM_TASKS]:
            if resource <= self.internal_state[-1][2]:
                possible = True
                break

        if self.num_step == MAX_STEP:
            done = True
            if 0 not in self.internal_state[:, 1]:
                info['DoneReasonType'] = DoneReasonType0.TYPE_3

            elif self.resource_of_all_tasks_selected <= SUM_RESOURCE_CAPACITY:
                info['DoneReasonType'] = DoneReasonType0.TYPE_2

            else:
                info['DoneReasonType'] = DoneReasonType0.TYPE_1

        elif not possible:
            done = True
            if 0 not in self.internal_state[:, 1]:
                info['DoneReasonType'] = DoneReasonType0.TYPE_3

            elif self.resource_of_all_tasks_selected <= SUM_RESOURCE_CAPACITY:
                info['DoneReasonType'] = DoneReasonType0.TYPE_2

            else:
                info['DoneReasonType'] = DoneReasonType0.TYPE_1

        else:
            done = False

        observation = self.observation()

        if done:
            reward = self.reward(done_type=info['DoneReasonType'])
        else:
            reward = self.reward(done_type=None)

        info['Actions sequence'] = self.actions_sequence
        info['Tasks selected'] = self.tasks_selected
        info['Resources allocated'] = self.resource_of_all_tasks_selected
        info['Limit'] = SUM_RESOURCE_CAPACITY
        info['internal_state'] = self.internal_state

        return observation, reward, done, info

    def reward(self, done_type=None):
        if done_type is None:  # Normal Step
            resource_efficiency_reward = 0.0
            mission_complete_reward = 0.0
            misbehavior_reward = 0.0

        elif done_type == DoneReasonType0.TYPE_1:  # Resource Limit Exceeded
            resource_efficiency_reward = 0.0
            mission_complete_reward = -1.0
            misbehavior_reward = 0.0

        elif done_type == DoneReasonType0.TYPE_2:  # Not Enough Resource or All Tasks Selected - [NORMAL]
            resource_efficiency_reward = 0.0
            mission_complete_reward = self.resource_of_all_tasks_selected / SUM_RESOURCE_CAPACITY
            misbehavior_reward = 0.0

        elif done_type == DoneReasonType0.TYPE_3:  # All Tasks Selected
            resource_efficiency_reward = 0.0
            mission_complete_reward = self.resource_of_all_tasks_selected / SUM_RESOURCE_CAPACITY
            misbehavior_reward = 0.0

        else:
            raise ValueError()

        return resource_efficiency_reward + mission_complete_reward + misbehavior_reward
