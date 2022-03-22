import gym
from gym import spaces
import enum
import numpy as np
import copy
from typing import Optional
import random

from a_configuration.a_base_config.a_environments.config_knapsack_problem import ConfigKnapsackProblem0

class DoneReasonType0(enum.Enum):
    TYPE_1 = "Weight Limit Exceeded"
    TYPE_2 = "Weight Remains"
    TYPE_3 = "All Item Selected"


class KnapsackProblemEnv(gym.Env):
    def __init__(self, config):
        self.NUM_ITEM = config.NUM_ITEM
        self.MAX_WEIGHT = config.MAX_WEIGHT

        self.MIN_WEIGHT_ITEM = config.MIN_WEIGHT_ITEM
        self.MAX_WEIGHT_ITEM = config.MAX_WEIGHT_ITEM

        self.MIN_VALUE_ITEM = config.MIN_VALUE_ITEM
        self.MAX_VALUE_ITEM = config.MAX_VALUE_ITEM

        self.INITIAL_ITEM_DISTRIBUTION_FIXED = config.INITIAL_ITEM_DISTRIBUTION_FIXED

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

    def get_initial_state(self):
        state = np.zeros(shape=(self.NUM_ITEM + 1, 4), dtype=int)

        for item_idx in range(self.NUM_ITEM):
            item_weight = np.random.randint(
                low=self.MIN_WEIGHT_ITEM,
                high=self.MAX_WEIGHT_ITEM,
                size=(1, 1)
            )
            item_value = np.random.randint(
                low=self.MIN_VALUE_ITEM,
                high=self.MAX_VALUE_ITEM,
                size=(1, 1)
            )
            state[item_idx][2] = item_weight
            state[item_idx][3] = item_value

        state[-1][2] = np.array(self.MAX_WEIGHT)

        return state

    def check_select_possible(self):
        possible = False

        for item in self.internal_state[self.num_step:self.NUM_ITEM]:
            weight = item[2]

            if weight <= self.internal_state[-1][2]:
                possible = True
                break

        return possible

    def observation(self):
        observation = copy.deepcopy(self.internal_state.flatten()) / self.MAX_WEIGHT
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
            value_of_all_items_selected_reward = self.value_of_all_items_selected / self.MAX_WEIGHT
            mission_complete_reward = 0.0
            misbehavior_reward = 0.0

        elif done_type == DoneReasonType0.TYPE_3:  # "All Item Selected"
            value_of_all_items_selected_reward = self.value_of_all_items_selected / self.MAX_WEIGHT
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

    def step(self, action_idx):
        self.actions_sequence.append(action_idx)
        info = dict()
        step_item_weight, step_item_value = self.internal_state[self.num_step][2:]
        #step_item_value = self.internal_state[self.num_step][3]

        if action_idx == 1:
            self.items_selected.append(self.num_step)
            self.weight_of_all_items_selected += step_item_weight
            self.value_of_all_items_selected += step_item_value
            self.internal_state[-1][2] -= step_item_weight

            self.internal_state[self.num_step][1] = 1
            self.internal_state[self.num_step][2:] = -1
            self.internal_state[-1][0] = self.weight_of_all_items_selected
            self.internal_state[-1][1] = self.value_of_all_items_selected

        possible = self.check_select_possible()

        done = False
        if self.num_step == self.NUM_ITEM - 1 or not possible:
            done = True

            if self.weight_of_all_items_selected > self.MAX_WEIGHT:
                info['DoneReasonType'] = DoneReasonType0.TYPE_1  # "Weight Limit Exceeded"
            elif 0 not in self.internal_state[:, 1]:
                info['DoneReasonType'] = DoneReasonType0.TYPE_3  # "All Item Selected"
            else:
                info['DoneReasonType'] = DoneReasonType0.TYPE_2  # "Weight Remains"

        else:
            self.internal_state[self.num_step][0] = 0
            self.num_step += 1
            self.internal_state[self.num_step][0] = 1

        observation = self.observation()

        if done:
            reward = self.reward(done_type=info['DoneReasonType'])
        else:
            reward = self.reward(done_type=None)

        info['Actions sequence'] = self.actions_sequence
        info['Tasks selected'] = self.items_selected
        info['Value'] = self.value_of_all_items_selected
        info['Weight'] = self.weight_of_all_items_selected
        info['internal_state'] = copy.deepcopy(self.internal_state)

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

    config = ConfigKnapsackProblem0()
    env = KnapsackProblemEnv(config)

    for i in range(2):
        observation, info = env.reset(return_info=True)
        done = False
        while not done:
            action = agent.get_action(observation)
            next_observation, reward, done, next_info = env.step(action)

            print("Observation: {0}, Action: {1}, next_observation: {2}, Reward: {3}, Done: {4}".format(
                info['internal_state'], action, next_info['internal_state'], reward, done
            ))
            observation = next_observation
            info = next_info


if __name__ == "__main__":
    run_env()