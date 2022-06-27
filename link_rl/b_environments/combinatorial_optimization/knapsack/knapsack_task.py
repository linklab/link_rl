import gym
from gym import spaces
import enum
import numpy as np
import copy
from typing import Optional
import random
import datetime as dt

from a_configuration.a_base_config.a_environments.combinatorial_optimization.config_knapsack import ConfigKnapsack1
from a_configuration.a_base_config.c_models.config_convolutional_models import Config1DConvolutionalModel, \
    Config2DConvolutionalModel
from a_configuration.a_base_config.c_models.config_linear_models import ConfigLinearModel
from a_configuration.a_base_config.config_parse import SYSTEM_USER_NAME, SYSTEM_COMPUTER_NAME
from b_environments.combinatorial_optimization.boto3_knapsack import upload_file
from b_environments.combinatorial_optimization.knapsack_gurobi import model_kp
from g_utils.types import ModelType


class DoneReasonType0(enum.Enum):
    TYPE_0 = "Selected same items"
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
        self.UPLOAD_PATH = 'knapsack_instances/Actions/link_solution'
        date = dt.datetime.now()
        date_str = '/' + str(date.year) + str(date.month) + str(date.day)
        user = SYSTEM_USER_NAME
        com = SYSTEM_COMPUTER_NAME
        self.UPLOAD_PATH = self.UPLOAD_PATH + date_str + user + com + '/link_solution' + str(
            config.INSTANCE_INDEX) + '.csv'

        self.TOTAL_VALUE_FOR_ALL_ITEMS = None

        self.INITIAL_ITEM_DISTRIBUTION_FIXED = True

        self.solution_found = [0.0]
        self.optimal_value = 0

        self.internal_state = None
        self.items_selected = None
        self.actions_sequence = None
        self.weight_of_all_items_selected = None
        self.value_of_all_items_selected = None
        self.num_step = None

        self.action_space = spaces.Discrete(self.NUM_ITEM)

        self.observation_space = spaces.Box(
            low=-1.0, high=1000.0,
            shape=(2*(self.NUM_ITEM + 4),)
        )

        if self.INITIAL_ITEM_DISTRIBUTION_FIXED:
            self.fixed_initial_internal_state = self.get_initial_state()

    # Last Row in State
    # 0: Always 0
    # 1: initially set to LIMIT_WEIGHT_KNAPSACK, and decrease by an item's weight whenever the item is selected
    # 2: initially set to 0, and increase by an item's weight whenever the item is selected
    # 3: initially set to 0, and increase by an item's value whenever the item is selected
    def get_initial_state(self):
        state = np.zeros(shape=(self.NUM_ITEM + 4, 2), dtype=float)

        for item_idx in range(self.NUM_ITEM):
            item_weight = np.random.randint(
                low=self.MIN_WEIGHT_ITEM, high=self.MAX_WEIGHT_ITEM, size=(1, 1)
            )
            item_value = np.random.randint(
                low=self.MIN_VALUE_ITEM, high=self.MAX_VALUE_ITEM, size=(1, 1)
            )
            state[item_idx+4][0] = item_value
            state[item_idx+4][1] = item_weight

            state[1][0] += item_value
            state[1][1] += item_weight

        state[0][0] = self.NUM_ITEM
        state[0][1] = self.LIMIT_WEIGHT_KNAPSACK

        values = state[4:, 0]
        weigths = state[4:, 1]
        items_selected, self.optimal_value = model_kp(self.LIMIT_WEIGHT_KNAPSACK, values, weigths, False)
        return state

    def observation(self):
        observation = copy.deepcopy(self.internal_state.flatten()) / self.LIMIT_WEIGHT_KNAPSACK
        return observation

    def reward(self, done_type=None):
        if done_type is None:  # Normal Step
            value_of_all_items_selected_reward = 0.0
            mission_complete_reward = 0.0
            misbehavior_reward = 0.0

        elif done_type == DoneReasonType0.TYPE_0:  # "selecte same item"
            value_of_all_items_selected_reward = 0.0
            mission_complete_reward = 0.0
            misbehavior_reward = -1.0

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


        self.TOTAL_VALUE_FOR_ALL_ITEMS = sum(self.internal_state[:, 0])
        self.items_selected = []
        self.actions_sequence = []
        self.weight_of_all_items_selected = 0
        self.value_of_all_items_selected = 0

        self.num_step = 0

        observation = self.observation()

        info = dict()
        info['internal_state'] = copy.deepcopy(self.internal_state)

        if return_info:
            return observation[None, :], info
        else:
            return observation[None, :]

    def check_future_select_possible(self):
        possible = False

        for item in self.internal_state[self.num_step + 3:-1]:
            weight = item[1]

            if weight != -1 and weight + self.internal_state[2][1] <= self.internal_state[0][1]:
                possible = True
                break

        return possible

    def step(self, action_idx):
        info = dict()
        step_item_value, step_item_weight = self.internal_state[action_idx + 3][:]

        if action_idx in self.items_selected:
            done = True
            info['DoneReasonType'] = DoneReasonType0.TYPE_0

        else:
            self.items_selected.append(action_idx)
            self.value_of_all_items_selected += step_item_value
            self.weight_of_all_items_selected += step_item_weight

            self.internal_state[2][0] += step_item_value
            self.internal_state[2][1] += step_item_weight

            possible = self.check_future_select_possible()

            self.internal_state[action_idx + 4][:] = -1

            if not possible:
                done = True

                if self.weight_of_all_items_selected > self.LIMIT_WEIGHT_KNAPSACK:
                    info['DoneReasonType'] = DoneReasonType0.TYPE_1  # "Weight Limit Exceeded"
                else:
                    info['DoneReasonType'] = DoneReasonType0.TYPE_2  # "Weight Remains"


            else:
                done = False

        self.internal_state[0][0] -= 1

        observation = self.observation()

        if done:
            reward = self.reward(done_type=info['DoneReasonType'])

            if info['DoneReasonType'] != DoneReasonType0.TYPE_0 or info['DoneReasonType'] != DoneReasonType0.TYPE_1:
                if self.solution_found[0] < self.value_of_all_items_selected:
                    self.solution_found[0] = self.value_of_all_items_selected
                    self.solution_found[1:] = self.items_selected

                    self.solution_found.append(round(self.solution_found[0] / self.optimal_value, 3))

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

        return observation[None, :], reward, done, info

def run_env():
    class Dummy_Agent:
        def get_action(self, observation):
            assert observation is not None
            available_action_ids = list(range(20))
            action_id = random.choice(available_action_ids)
            return action_id

    print("START RUN!!!")
    agent = Dummy_Agent()

    config = ConfigKnapsack1()
    env = KnapsackEnv(config)

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
