import gym
from gym import spaces
import enum
import numpy as np
import copy
from typing import Optional
import random
import datetime as dt

from link_rl.a_configuration.a_base_config.c_models.config_linear_models import ConfigLinearModel
from link_rl.a_configuration.a_base_config.config_parse import SYSTEM_USER_NAME, SYSTEM_COMPUTER_NAME
from link_rl.b_environments.combinatorial_optimization.knapsack.boto3_knapsack import load_instance, upload_file, load_solution
from link_rl.b_environments.combinatorial_optimization.knapsack.knapsack_gurobi import model_kp

STATIC_INITIAL_STATE_50 = np.asarray([
    [50.000, 200.000],
    [508.000, 499.000],
    [0.000, 0.000], #sum2
    [0.000, 0.000], #indicator3
    [12.000, 12.000],
    [19.000, 11.000],
    [11.000, 4.000],
    [14.000, 3.000],
    [11.000, 4.000],
    [2.000, 19.000],
    [7.000, 10.000],
    [11.000, 1.000],
    [17.000, 10.000],
    [17.000, 6.000],
    [6.000, 18.000],
    [13.000, 5.000],
    [2.000, 2.000],
    [1.000, 9.000],
    [13.000, 19.000],
    [15.000, 8.000],
    [19.000, 8.000],
    [7.000, 12.000],
    [15.000, 13.000],
    [10.000, 2.000],
    [7.000, 19.000],
    [13.000, 3.000],
    [2.000, 8.000],
    [15.000, 10.000],
    [15.000, 8.000],
    [6.000, 14.000],
    [3.000, 17.000],
    [2.000, 15.000],
    [6.000, 4.000],
    [16.000, 12.000],
    [11.000, 14.000],
    [15.000, 11.000],
    [12.000, 1.000],
    [4.000, 18.000],
    [18.000, 15.000],
    [6.000, 4.000],
    [16.000, 6.000],
    [2.000, 3.000],
    [13.000, 16.000],
    [12.000, 12.000],
    [1.000, 11.000],
    [5.000, 11.000],
    [18.000, 19.000],
    [3.000, 17.000],
    [5.000, 15.000],
    [18.000, 7.000],
    [18.000, 1.000],
    [8.000, 8.000],
    [3.000, 14.000],
    [13.000, 10.000]
])

STATIC_INITIAL_STATE_50_OPTIMAL = 385


class DoneReasonType0(enum.Enum):
    TYPE_0 = "Selected same items"
    TYPE_1 = "Weight Limit Exceeded"
    TYPE_2 = "Weight Remains"
    TYPE_3 = "All Item Selected"


class KnapsackEnv(gym.Env):
    def __init__(self, config):
        self.config = config
        self.NUM_ITEM = config.NUM_ITEM
        self.LIMIT_WEIGHT_KNAPSACK = config.LIMIT_WEIGHT_KNAPSACK

        self.MIN_WEIGHT_ITEM = config.MIN_WEIGHT_ITEM
        self.MAX_WEIGHT_ITEM = config.MAX_WEIGHT_ITEM

        self.MIN_VALUE_ITEM = config.MIN_VALUE_ITEM
        self.MAX_VALUE_ITEM = config.MAX_VALUE_ITEM

        self.INITIAL_ITEM_DISTRIBUTION_FIXED = config.INITIAL_ITEM_DISTRIBUTION_FIXED

        self.FILE_PATH = config.FILE_PATH
        self.UPLOAD_PATH = config.UPLOAD_PATH
        self.OPTIMAL_PATH = config.OPTIMAL_PATH
        self.INSTANCE_INDEX = config.INSTANCE_INDEX
        self.SORTING_TYPE = config.SORTING_TYPE

        self.last_ep_solution_found = [0]
        self.optimal_value = 0

        self.internal_state = None
        self.items_selected = None
        self.actions_sequence = None
        self.last_ep_weight_of_all_items_selected = None
        self.last_ep_value_of_all_items_selected = None
        self.num_step = None

        self.action_space = spaces.Discrete(self.NUM_ITEM)

        if isinstance(config.MODEL_PARAMETER, ConfigLinearModel):
            self.observation_space = spaces.Box(
                low=-1.0, high=1000.0,
                shape=((self.NUM_ITEM + 4) * 2,)
            )
        elif isinstance(config.MODEL_PARAMETER, Config1DConvolutionalModel):
            self.observation_space = spaces.Box(
                low=-1.0, high=1000.0,
                shape=(self.NUM_ITEM + 4, 2)
            )
        else:
            raise ValueError()

        if self.INITIAL_ITEM_DISTRIBUTION_FIXED:
            self.fixed_initial_internal_state = self.get_initial_state()

    # Last Row in State
    # 0: Always 0
    # 1: initially set to LIMIT_WEIGHT_KNAPSACK, and decrease by an item's weight whenever the item is selected
    # 2: initially set to 0, and increase by an item's weight whenever the item is selected
    # 3: initially set to 0, and increase by an item's value whenever the item is selected
    def get_initial_state(self):
        if self.config.STATIC_INITIAL_STATE_50:
            state = copy.deepcopy(STATIC_INITIAL_STATE_50)
            self.LIMIT_WEIGHT_KNAPSACK = state[0][1]
            self.optimal_value = STATIC_INITIAL_STATE_50_OPTIMAL
            print("*** STATIC_INITIAL_STATE_50 is used!!! ***")

        elif self.FILE_PATH:
            self.FILE_PATH = self.FILE_PATH + '/instance' + str(self.INSTANCE_INDEX) + '.csv'
            data = load_instance('linklab', self.FILE_PATH)

            state = data

            for item_idx in range(self.NUM_ITEM):
                state[1][0] += state[item_idx + 4][0]
                state[1][1] += state[item_idx + 4][1]

            self.LIMIT_WEIGHT_KNAPSACK = state[0][1]
            state[0][0] = self.NUM_ITEM

        else:
            state = np.zeros(shape=(self.NUM_ITEM + 4, 2), dtype=float)

            for item_idx in range(self.NUM_ITEM):
                item_weight = np.random.randint(
                    low=self.MIN_WEIGHT_ITEM, high=self.MAX_WEIGHT_ITEM, size=(1, 1)
                )
                item_value = np.random.randint(
                    low=self.MIN_VALUE_ITEM, high=self.MAX_VALUE_ITEM, size=(1, 1)
                )
                state[item_idx + 4][0] = item_value
                state[item_idx + 4][1] = item_weight

                state[1][0] += item_value
                state[1][1] += item_weight

            state[0][0] = self.NUM_ITEM
            state[0][1] = self.LIMIT_WEIGHT_KNAPSACK

        if self.OPTIMAL_PATH:
            self.OPTIMAL_PATH = self.OPTIMAL_PATH + '/solution' + str(self.INSTANCE_INDEX) + '.csv'
            self.optimal_value = load_solution('linklab', self.OPTIMAL_PATH)
        else:
            values = state[4:, 0]
            weigths = state[4:, 1]
            items_selected, self.optimal_value = model_kp(self.LIMIT_WEIGHT_KNAPSACK, values, weigths, False)

        if self.UPLOAD_PATH:
            date = dt.datetime.now()
            date_str = '/' + str(date.year) + str(date.month) + str(date.day)
            user = SYSTEM_USER_NAME
            com = SYSTEM_COMPUTER_NAME
            self.UPLOAD_PATH = self.UPLOAD_PATH + date_str + user + com + '/link_solution' + str(
                self.INSTANCE_INDEX) + '.csv'
        else:
            self.UPLOAD_PATH = 'knapsack_instances'
            date = dt.datetime.now()
            date_str = '/' + str(date.year) + str(date.month) + str(date.day)
            user = SYSTEM_USER_NAME
            com = SYSTEM_COMPUTER_NAME
            self.UPLOAD_PATH = self.UPLOAD_PATH + date_str + user + com + '/link_solution' + str(
                self.INSTANCE_INDEX) + '.csv'

        return state

    def state_sorting(self, state, start, end):

        if self.SORTING_TYPE == 1: #Value Per Weight
            if start >= end:
                return

            pivot = start
            left = start + 1
            right = end

            while left <= right:
                while left <= end and (state[left][0] / state[left][1]) <= (state[pivot][0] / state[pivot][1]):
                    left += 1

                while right > start and (state[right][0] / state[right][1]) >= (state[pivot][0] / state[pivot][0]):
                    right -= 1

                if left > right:
                    state[[right, pivot]] = state[[pivot, right]]
                else:
                    state[[left, right]] = state[[right, left]]

            self.state_sorting(state, start, right - 1)
            self.state_sorting(state, right + 1, end)

        elif self.SORTING_TYPE == 2: #Value
            if start >= end:
                return

            pivot = start
            left = start + 1
            right = end

            while left <= right:
                while left <= end and state[left][0] <= state[pivot][0]:
                    left += 1

                while right > start and state[right][0] >= state[pivot][0]:
                    right -= 1

                if left > right:
                    state[[right, pivot]] = state[[pivot, right]]
                else:
                    state[[left, right]] = state[[right, left]]

            self.state_sorting(state, start, right - 1)
            self.state_sorting(state, right + 1, end)

        elif self.SORTING_TYPE == 3: #weight
            if start >= end:
                return

            pivot = start
            left = start + 1
            right = end

            while left <= right:
                while left <= end and state[left][1] <= state[pivot][1]:
                    left += 1

                while right > start and state[right][1] >= state[pivot][1]:
                    right -= 1

                if left > right:
                    state[[right, pivot]] = state[[pivot, right]]
                else:
                    state[[left, right]] = state[[right, left]]

            self.state_sorting(state, start, right - 1)
            self.state_sorting(state, right + 1, end)

        elif self.SORTING_TYPE is None:
            pass

        else:
            raise ValueError()

    def observation(self):
        if isinstance(self.config.MODEL_PARAMETER, ConfigLinearModel):
            observation = copy.deepcopy(self.internal_state.flatten()) / self.LIMIT_WEIGHT_KNAPSACK
        elif isinstance(self.config.MODEL_PARAMETER, Config1DConvolutionalModel):
            observation = copy.deepcopy(self.internal_state) / self.LIMIT_WEIGHT_KNAPSACK
        else:
            raise ValueError()
        return observation

    def reward(self, done_type=None):
        if done_type is None:  # Normal Step
            value_of_all_items_selected_reward = 0.0
            mission_complete_reward = 0.0
            misbehavior_reward = 0.0

        elif done_type == DoneReasonType0.TYPE_0:  # "Selected Same Item"
            value_of_all_items_selected_reward = 0.0
            mission_complete_reward = 0.0
            misbehavior_reward = -1.0

        elif done_type == DoneReasonType0.TYPE_1:  # "Weight Limit Exceeded"
            value_of_all_items_selected_reward = 0.0
            mission_complete_reward = 0.0
            misbehavior_reward = -1.0

        elif done_type == DoneReasonType0.TYPE_2:  # "Weight Remains"
            value_of_all_items_selected_reward = self.last_ep_value_of_all_items_selected / self.TOTAL_VALUE_FOR_ALL_ITEMS
            mission_complete_reward = 0.0
            misbehavior_reward = 0.0

        elif done_type == DoneReasonType0.TYPE_3:  # "All Item Selected"
            value_of_all_items_selected_reward = self.last_ep_value_of_all_items_selected / self.TOTAL_VALUE_FOR_ALL_ITEMS
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

        if self.SORTING_TYPE:
            self.state_sorting(self.internal_state, 4, self.NUM_ITEM + 3)

        self.TOTAL_VALUE_FOR_ALL_ITEMS = sum(self.internal_state[:, 0])
        self.items_selected = []
        self.actions_sequence = []
        self.last_ep_weight_of_all_items_selected = 0
        self.last_ep_value_of_all_items_selected = 0

        observation = self.observation()
        info = dict()
        info['internal_state'] = copy.deepcopy(self.internal_state)

        if return_info:
            return observation, info
        else:
            return observation

    def check_future_select_possible(self):
        possible = False

        for item in self.internal_state[4:-1]:
            weight = item[1]

            if weight != -1 and weight + self.internal_state[2][1] <= self.internal_state[0][1]:
                possible = True
                break

        return possible

    def step(self, action_idx):
        info = dict()
        step_item_value, step_item_weight = self.internal_state[action_idx + 4][:]

        if action_idx in self.actions_sequence:
            done = True
            info['DoneReasonType'] = DoneReasonType0.TYPE_0
            self.actions_sequence.append(action_idx)

        else:
            self.actions_sequence.append(action_idx)
            self.items_selected.append(action_idx)
            self.last_ep_value_of_all_items_selected += step_item_value
            self.last_ep_weight_of_all_items_selected += step_item_weight

            self.internal_state[2][0] += step_item_value
            self.internal_state[2][1] += step_item_weight

            possible = self.check_future_select_possible()

            self.internal_state[action_idx + 4][:] = -1

            if not possible:
                done = True

                if self.last_ep_weight_of_all_items_selected > self.LIMIT_WEIGHT_KNAPSACK:
                    info['DoneReasonType'] = DoneReasonType0.TYPE_1  # "Weight Limit Exceeded"
                else:
                    info['DoneReasonType'] = DoneReasonType0.TYPE_2  # "Weight Remains"
            else:
                done = False

        self.internal_state[0][0] -= 1

        observation = self.observation()

        if done:
            reward = self.reward(done_type=info['DoneReasonType'])

            if info['DoneReasonType'] != DoneReasonType0.TYPE_0 and info['DoneReasonType'] != DoneReasonType0.TYPE_1:
                if self.last_ep_solution_found[0] < self.last_ep_value_of_all_items_selected:
                    self.last_ep_solution_found[0] = self.last_ep_value_of_all_items_selected
                    self.last_ep_solution_found[1:] = self.items_selected

                    self.last_ep_solution_found.append(round(self.last_ep_solution_found[0] / self.optimal_value, 3))

                    date = dt.datetime.now()
                    self.last_ep_solution_found.append((str(date.month) + '/' + str(date.day) + ' ' + str(
                        date.hour) + ' : ' + str(date.minute) + ' : ' + str(date.second)))

                    if self.UPLOAD_PATH:
                        upload_file('linklab', self.last_ep_solution_found, self.UPLOAD_PATH)
        else:
            reward = self.reward(done_type=None)

        info['Actions sequence'] = self.actions_sequence
        info['Items selected'] = self.items_selected
        info['Value'] = self.last_ep_value_of_all_items_selected
        info['Weight'] = self.last_ep_weight_of_all_items_selected
        info['internal_state'] = copy.deepcopy(self.internal_state)
        info['last_ep_solution_found'] = self.last_ep_solution_found

        return observation, reward, done, info

#Random Instance Test
from link_rl.a_configuration.a_base_config.a_environments.combinatorial_optimization.knapsack.config_knapsack import \
    ConfigKnapsack0RandomTest
config = ConfigKnapsack0RandomTest()

#Load Instance Test
from link_rl.a_configuration.a_base_config.a_environments.combinatorial_optimization.knapsack.config_knapsack import \
    ConfigKnapsack0LoadTest
config = ConfigKnapsack0LoadTest()

#Static Instance Test
from link_rl.a_configuration.a_base_config.a_environments.combinatorial_optimization.knapsack.config_knapsack import \
    ConfigKnapsack0StaticTest
config = ConfigKnapsack0StaticTest()

from link_rl.a_configuration.a_base_config.c_models.config_convolutional_models import Config1DConvolutionalModel
config.MODEL_PARAMETER = Config1DConvolutionalModel(config.MODEL_TYPE)

def run_env(config):
    class Dummy_Agent:
        def get_action(self, observation):
            assert observation is not None
            available_action_ids = list(range(config.NUM_ITEM))
            action_id = random.choice(available_action_ids)
            return action_id

    print("START RUN!!!")
    agent = Dummy_Agent()
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
    run_env(config)
