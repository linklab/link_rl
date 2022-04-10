import csv

import gym
from gym import spaces
import enum
import numpy as np
import copy
from typing import Optional
import random

from a_configuration.a_base_config.a_environments.combinatorial_optimization.config_knapsack import ConfigKnapsack0
from b_environments.combinatorial_optimization.boto3_knapsack import load_instance, upload_file


class DoneReasonType0(enum.Enum):
    TYPE_1 = "Weight Limit Exceeded"
    TYPE_2 = "Weight Remains"
    TYPE_3 = "All Item Selected"


class KnapsackEnv(gym.Env):
    def __init__(self, config):
        self.NUM_ITEM = config.NUM_ITEM
        self.LIMIT_WEIGHT_KNAPSACK = config.LIMIT_WEIGHT_KNAPSACK

        self.MIN_WEIGHT_ITEM = config.MIN_WEIGHT_ITEM
        self.MAX_WEIGHT_ITEM = config.MAX_WEIGHT_ITEM

        self.MIN_VALUE_ITEM = config.MIN_VALUE_ITEM
        self.MAX_VALUE_ITEM = config.MAX_VALUE_ITEM

        self.TOTAL_VALUE_FOR_ALL_ITEMS = None

        self.INITIAL_ITEM_DISTRIBUTION_FIXED = config.INITIAL_ITEM_DISTRIBUTION_FIXED

        self.FILE_PATH = config.FILE_PATH
        self.UPLOAD_PATH = config.UPLOAD_PATH

        self.solution_found = [0]

        self.internal_state = None
        self.items_selected = None
        self.actions_sequence = None
        self.weight_of_all_items_selected = None
        self.value_of_all_items_selected = None
        self.num_step = None

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(
            low=-1.0, high=1000.0,
            shape=((self.NUM_ITEM + 1) * 4,)
        )

        if self.INITIAL_ITEM_DISTRIBUTION_FIXED:
            self.fixed_initial_internal_state = self.get_initial_state()

    # Last Row in State
    # 0: Always 0
    # 1: initially set to LIMIT_WEIGHT_KNAPSACK, and decrease by an item's weight whenever the item is selected
    # 2: initially set to 0, and increase by an item's weight whenever the item is selected
    # 3: initially set to 0, and increase by an item's value whenever the item is selected
    def get_initial_state(self):
        state = np.zeros(shape=(self.NUM_ITEM + 1, 4), dtype=float)

        if self.FILE_PATH:
            data = load_instance('linklab', self.FILE_PATH)

            state = data
        else:
            for item_idx in range(self.NUM_ITEM):
                item_weight = np.random.randint(
                    low=self.MIN_WEIGHT_ITEM, high=self.MAX_WEIGHT_ITEM, size=(1, 1)
                )
                item_value = np.random.randint(
                    low=self.MIN_VALUE_ITEM, high=self.MAX_VALUE_ITEM, size=(1, 1)
                )
                state[item_idx][2] = item_value
                state[item_idx][3] = item_weight

            state[-1][1] = np.array(self.LIMIT_WEIGHT_KNAPSACK)

        # for item_idx in range(self.NUM_ITEM):
        #     state[item_idx][2] = data[item_idx][1]
        #     state[item_idx][3] = data[item_idx][0]
        #
        # state[-1][1] = data[self.NUM_ITEM][1]
        return state

    def observation(self):
        observation = copy.deepcopy(self.internal_state.flatten()) / self.LIMIT_WEIGHT_KNAPSACK
        return observation

    def reward(self, done_type=None):
        if done_type is None:  # Normal Step
            value_of_all_items_selected_reward = 0.0
            mission_complete_reward = 0.0
            misbehavior_reward = 0.0

        elif done_type == DoneReasonType0.TYPE_1:  # "Weight Limit Exceeded"
            value_of_all_items_selected_reward = 0.0
            mission_complete_reward = 0.0
            misbehavior_reward = -1.0

        elif done_type == DoneReasonType0.TYPE_2:  # "Weight Remains"
            value_of_all_items_selected_reward = self.value_of_all_items_selected / self.TOTAL_VALUE_FOR_ALL_ITEMS
            mission_complete_reward = 0.0
            misbehavior_reward = 0.0

        elif done_type == DoneReasonType0.TYPE_3:  # "All Item Selected"
            value_of_all_items_selected_reward = self.value_of_all_items_selected / self.TOTAL_VALUE_FOR_ALL_ITEMS
            mission_complete_reward = 1.0
            misbehavior_reward = 0.0

        else:
            raise ValueError()

        return value_of_all_items_selected_reward + mission_complete_reward + misbehavior_reward

    def reset(self, *, seed: Optional[int] = None, return_info: bool = False, options: Optional[dict] = None,):
        if self.INITIAL_ITEM_DISTRIBUTION_FIXED:
            assert self.fixed_initial_internal_state is not None
            self.internal_state = copy.deepcopy(self.fixed_initial_internal_state)
        else:
            self.internal_state = self.get_initial_state()

        self.TOTAL_VALUE_FOR_ALL_ITEMS = sum(self.internal_state[:, 2])
        self.items_selected = []
        self.actions_sequence = []
        self.weight_of_all_items_selected = 0
        self.value_of_all_items_selected = 0

        self.num_step = 0
        self.internal_state[self.num_step][0] = 1

        observation = self.observation()
        info = dict()
        info['internal_state'] = copy.deepcopy(self.internal_state)

        if return_info:
            return observation, info
        else:
            return observation

    def check_future_select_possible(self):
        possible = False

        for item in self.internal_state[self.num_step + 1: self.NUM_ITEM]:
            weight = item[3]

            if weight <= self.internal_state[-1][1]:
                possible = True
                break

        return possible

    def step(self, action_idx):
        self.actions_sequence.append(action_idx)
        info = dict()
        step_item_value, step_item_weight = self.internal_state[self.num_step][2:]

        if action_idx == 1:
            self.items_selected.append(self.num_step)
            self.value_of_all_items_selected += step_item_value
            self.weight_of_all_items_selected += step_item_weight

            self.internal_state[self.num_step][1] = 1
            self.internal_state[self.num_step][2:] = -1

            self.internal_state[-1][1] -= step_item_weight
            self.internal_state[-1][3] = self.value_of_all_items_selected
            self.internal_state[-1][2] = self.weight_of_all_items_selected

        possible = self.check_future_select_possible()

        done = False
        if self.num_step == self.NUM_ITEM - 1 or not possible:
            done = True

            if 0 not in self.internal_state[:, 1]:
                info['DoneReasonType'] = DoneReasonType0.TYPE_3  # "All Item Selected"
            elif self.weight_of_all_items_selected > self.LIMIT_WEIGHT_KNAPSACK:
                info['DoneReasonType'] = DoneReasonType0.TYPE_1  # "Weight Limit Exceeded"
            else:
                info['DoneReasonType'] = DoneReasonType0.TYPE_2  # "Weight Remains"

            self.internal_state[self.num_step][0] = 0
            self.internal_state[-1][0] = 1
        else:
            self.internal_state[self.num_step][0] = 0
            self.num_step += 1
            self.internal_state[self.num_step][0] = 1

        observation = self.observation()

        if done:
            reward = self.reward(done_type=info['DoneReasonType'])
            if self.solution_found[0] < self.value_of_all_items_selected:
                self.solution_found[0] = self.value_of_all_items_selected
                self.solution_found[1:] = self.items_selected
                if self.UPLOAD_PATH:
                    upload_file('linklab', self.solution_found, self.UPLOAD_PATH)
        else:
            reward = self.reward(done_type=None)

        info['Actions sequence'] = self.actions_sequence
        info['Items selected'] = self.items_selected
        info['Value'] = self.value_of_all_items_selected
        info['Weight'] = self.weight_of_all_items_selected
        info['internal_state'] = copy.deepcopy(self.internal_state)
        info['solution_found'] = self.solution_found

        return observation, reward, done, info


def run_env():
    class Dummy_Agent:
        def get_action(self, observation):
            assert observation is not None
            available_action_ids = [0, 1]
            action_id = random.choice(available_action_ids)
            return action_id

    print("START RUN!!!")
    agent = Dummy_Agent()

    config = ConfigKnapsack0()
    config.NUM_ITEM = 50
    env = KnapsackEnv(config)
    # config.FILE_PATH = 'knapsack_instances/RI/instances/n_300_r_600/instance0.csv'
    # config.UPLOAD_PATH = 'knapsack_instances/RI/link_solution/n_300_r_600/instance0.csv'

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