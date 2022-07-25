import gym
from gym import spaces
import enum
import numpy as np
import copy
from typing import Optional
import random
import datetime as dt
import plotly.graph_objects as go

from link_rl.a_configuration.a_base_config.c_models.config_convolutional_models import Config1DConvolutionalModel
from link_rl.a_configuration.a_base_config.c_models.config_linear_models import ConfigLinearModel
from link_rl.a_configuration.a_base_config.c_models.config_recurrent_convolutional_models import \
    ConfigRecurrent1DConvolutionalModel
from link_rl.a_configuration.a_base_config.c_models.config_recurrent_linear_models import ConfigRecurrentLinearModel
from link_rl.a_configuration.a_base_config.config_parse import SYSTEM_USER_NAME, SYSTEM_COMPUTER_NAME
from link_rl.b_environments.combinatorial_optimization.knapsack.boto3_knapsack import load_instance, upload_file, load_solution
from link_rl.b_environments.combinatorial_optimization.knapsack.knapsack_gurobi import model_kp
from link_rl.h_utils.commons import set_config
from link_rl.h_utils.stats import CustomEnvStat, CustomEnvComparisonStat
from link_rl.h_utils.types import HerConstant

STATIC_ITEMS_50 = np.asarray([
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

STATIC_INITIAL_STATE_50 = np.asarray([
    [50.000, 200.000],
    [508.000, 499.000],
    [0.000, 0.000], #sum2
    [0.000, 0.000], #indicator3
])

STATIC_INITIAL_STATE_50_OPTIMAL = 385


class DoneReasonType0(enum.Enum):
    TYPE_0 = "Selected Same items"
    TYPE_1 = "Weight Limit Exceeded"
    TYPE_2 = "Weight Remains"
    TYPE_3 = "Goal Achieved"


class KnapsackEnv(gym.Env):
    def __init__(self, config):
        self.config = config

        if self.config.STATIC_INITIAL_STATE_50:
            assert self.config.INITIAL_ITEM_DISTRIBUTION_FIXED

        self.weight_of_all_items_selected = None
        self.value_of_all_items_selected = None

        self.last_ep_weight_of_all_items_selected = None
        self.last_ep_value_of_all_items_selected = None
        self.last_ep_solution_found = [0]
        self.last_ep_simple_solution_found = None
        self.last_ep_ratio = None

        self.optimal_value = 0

        self.internal_state = None
        self.items_selected = None
        self.actions_sequence = None

        self.num_step = None
        self.episodes = 0

        if self.config.STRATEGY == 1:
            self.action_space = spaces.Discrete(self.config.NUM_ITEM)
        else:
            self.action_space = spaces.Discrete(2)

        if isinstance(config.MODEL_PARAMETER, (ConfigLinearModel, ConfigRecurrentLinearModel)):
            self.observation_space = spaces.Box(
                low=-1.0, high=1000.0,
                shape=((self.config.NUM_ITEM + 5) * 2,) if self.config.USE_HER else ((self.config.NUM_ITEM + 4) * 2,)
            )
        elif isinstance(config.MODEL_PARAMETER, (Config1DConvolutionalModel, ConfigRecurrent1DConvolutionalModel)):
            self.observation_space = spaces.Box(
                low=-1.0, high=1000.0,
                shape=(self.config.NUM_ITEM + 5, 2) if self.config.USE_HER else (self.config.NUM_ITEM + 4, 2)
            )
        else:
            raise ValueError()

        if self.config.INITIAL_ITEM_DISTRIBUTION_FIXED:
            self.fixed_initial_internal_state = self.get_initial_state()

        if self.config.USE_HER:
            self.current_goal = 0.0

    def sort_items(self, items):
        if self.config.SORTING_TYPE is not None and self.config.SORTING_TYPE == 1:    # Value Per Weight
            value_per_weights = np.expand_dims((items[:, 0] / items[:, 1]), axis=1)
            new_items = np.hstack([items, value_per_weights])
            new_items = np.asarray(sorted(new_items, key=lambda x: x[2], reverse=True))
            new_items = np.delete(new_items, 2, 1)
            return new_items
        elif self.config.SORTING_TYPE is not None and self.config.SORTING_TYPE == 2:  # Value
            new_items = np.asarray(sorted(items, key=lambda x: x[0], reverse=True))
            return new_items
        elif self.config.SORTING_TYPE is not None and self.config.SORTING_TYPE == 3:  # weight
            new_items = np.asarray(sorted(items, key=lambda x: x[1], reverse=True))
            return new_items
        else:
            return items

    # Last Row in State
    # 0: Always 0
    # 1: initially set to LIMIT_WEIGHT_KNAPSACK, and decrease by an item's weight whenever the item is selected
    # 2: initially set to 0, and increase by an item's weight whenever the item is selected
    # 3: initially set to 0, and increase by an item's value whenever the item is selected
    def get_initial_state(self):
        if self.config.STATIC_INITIAL_STATE_50:
            items = copy.deepcopy(STATIC_ITEMS_50)
            items = self.sort_items(items)

            state = copy.deepcopy(STATIC_INITIAL_STATE_50)
            state = np.vstack([state, items])

            self.config.LIMIT_WEIGHT_KNAPSACK = state[0][1]
            self.optimal_value = STATIC_INITIAL_STATE_50_OPTIMAL
            print("*** STATIC_INITIAL_STATE_50 is used!!! ***")

        elif self.config.INITIAL_STATE_FILE_PATH:
            self.config.INITIAL_STATE_FILE_PATH = self.config.INITIAL_STATE_FILE_PATH + '/instance' + str(self.config.INSTANCE_INDEX) + '.csv'
            items, self.config.LIMIT_WEIGHT_KNAPSACK = load_instance('linklab', self.config.INITIAL_STATE_FILE_PATH)

            items = self.sort_items(np.asarray(items))

            state = np.zeros(shape=(self.config.NUM_ITEM + 4, 2), dtype=float)

            for item_idx in range(self.config.NUM_ITEM):
                state[item_idx + 4][0] = items[item_idx][0]
                state[item_idx + 4][1] = items[item_idx][1]
                state[1][0] += items[item_idx][0]
                state[1][1] += items[item_idx][1]

            state[0][0] = self.config.NUM_ITEM
            state[0][1] = self.config.LIMIT_WEIGHT_KNAPSACK

        else:
            state = np.zeros(shape=(self.config.NUM_ITEM + 4, 2), dtype=float)

            items = []
            for item_idx in range(self.config.NUM_ITEM):
                item_value = np.random.randint(
                    low=self.config.MIN_VALUE_ITEM, high=self.config.MAX_VALUE_ITEM, size=(1, 1)
                )
                item_weight = np.random.randint(
                    low=self.config.MIN_WEIGHT_ITEM, high=self.config.MAX_WEIGHT_ITEM, size=(1, 1)
                )
                items.append([item_value, item_weight])

            items = self.sort_items(np.asarray(items))

            for item_idx in range(self.config.NUM_ITEM):
                state[item_idx + 4][0] = items[item_idx][0]
                state[item_idx + 4][1] = items[item_idx][1]
                state[1][0] += items[item_idx][0]
                state[1][1] += items[item_idx][1]

            state[0][0] = self.config.NUM_ITEM
            state[0][1] = self.config.LIMIT_WEIGHT_KNAPSACK

        if self.config.OPTIMAL_PATH:
            self.config.OPTIMAL_PATH = self.config.OPTIMAL_PATH + '/solution' + str(self.config.INSTANCE_INDEX) + '.csv'
            self.optimal_value = load_solution('linklab', self.config.OPTIMAL_PATH)
        else:
            values = state[4:, 0]
            weights = state[4:, 1]
            items_selected, self.optimal_value = model_kp(self.config.LIMIT_WEIGHT_KNAPSACK, values, weights, False)

        if self.config.UPLOAD_PATH:
            date = dt.datetime.now()
            date_str = '/' + str(date.year) + str(date.month) + str(date.day)
            user = SYSTEM_USER_NAME
            com = SYSTEM_COMPUTER_NAME
            self.config.UPLOAD_PATH = \
                self.config.UPLOAD_PATH + date_str + user + com + '/link_solution' + str(self.config.INSTANCE_INDEX) + '.csv'
        else:
            self.config.UPLOAD_PATH = 'knapsack_instances'
            date = dt.datetime.now()
            date_str = '/' + str(date.year) + str(date.month) + str(date.day)
            user = SYSTEM_USER_NAME
            com = SYSTEM_COMPUTER_NAME
            self.config.UPLOAD_PATH = \
                self.config.UPLOAD_PATH + date_str + user + com + '/link_solution' + str(self.config.INSTANCE_INDEX) + '.csv'

        return state

    def observation(self):
        if isinstance(self.config.MODEL_PARAMETER, (ConfigLinearModel, ConfigRecurrentLinearModel)):
            observation = copy.deepcopy(self.internal_state.flatten()) / self.config.LIMIT_WEIGHT_KNAPSACK
        elif isinstance(self.config.MODEL_PARAMETER, (Config1DConvolutionalModel, ConfigRecurrent1DConvolutionalModel)):
            observation = copy.deepcopy(self.internal_state) / self.config.LIMIT_WEIGHT_KNAPSACK
        else:
            raise ValueError()
        return observation

    def print_knapsack_details_at_episode_end(self, info, ratio):
        details = "[NEW EPISODE (STEPS: {0:4})] NUM ITEMS: {1}, LIMIT_WEIGHT_KNAPSACK: {2}, " \
                  "TOTAL_VALUE_FOR_ALL_ITEMS: {3:5.1f}, ITEM VALUES SELECTED: {4:5.1f} (OPTIMAL_VALUE: {5:5.1f}, " \
                  "RATIO: {6:4.2f}), ITEM WEIGHTS SELECTED: {7:5.1f}, DONE REASON: {8}".format(
            self.num_step, self.config.NUM_ITEM, self.config.LIMIT_WEIGHT_KNAPSACK, self.TOTAL_VALUE_FOR_ALL_ITEMS,
            self.last_ep_value_of_all_items_selected, self.optimal_value,
            ratio, self.last_ep_weight_of_all_items_selected, info['DoneReasonType'].value
        )
        print(details)

    def reset(self, *, seed: Optional[int] = None, return_info: bool = False, options: Optional[dict] = None):
        if self.config.INITIAL_ITEM_DISTRIBUTION_FIXED:
            assert self.fixed_initial_internal_state is not None
            self.internal_state = copy.deepcopy(self.fixed_initial_internal_state)
        else:
            self.internal_state = self.get_initial_state()

        # if self.config.SORTING_TYPE is not None:
        #     self.state_sorting(self.internal_state, 4, self.config.NUM_ITEM + 3)

        self.TOTAL_VALUE_FOR_ALL_ITEMS = sum(self.internal_state[4:, 0])

        self.items_selected = []
        self.actions_sequence = []

        self.weight_of_all_items_selected = 0
        self.value_of_all_items_selected = 0

        self.num_step = 0

        if self.config.STRATEGY == 1:   # Large Action Space
            self.internal_state[3][0] = 0
            self.internal_state[3][1] = 0
        else:                           # Small Action Space
            self.internal_state[3][0] = self.internal_state[4][0]
            self.internal_state[3][1] = self.internal_state[4][1]

        info = dict()

        if self.config.USE_HER:
            # self.current_goal = random.randint(1, self.TOTAL_VALUE_FOR_ALL_ITEMS)
            self.current_goal = self.TOTAL_VALUE_FOR_ALL_ITEMS
            goal_array = np.asarray([self.current_goal, self.current_goal])
            self.internal_state = np.vstack([self.internal_state, goal_array])
            info[HerConstant.ACHIEVED_GOAL] = self.current_goal
            info[HerConstant.DESIRED_GOAL] = self.current_goal

        info['internal_state'] = copy.deepcopy(self.internal_state)

        info['last_ep_value_of_all_items_selected'] = self.last_ep_value_of_all_items_selected
        info['last_ep_weight_of_all_items_selected'] = self.last_ep_weight_of_all_items_selected
        info['last_ep_solution_found'] = self.last_ep_solution_found
        info['last_ep_simple_solution_found'] = self.last_ep_simple_solution_found
        info['last_ep_ratio'] = self.last_ep_ratio

        self.episodes += 1

        observation = self.observation()

        if return_info:
            return observation, info
        else:
            return observation

    def step(self, action_idx):
        info = dict()

        if self.config.STRATEGY == 1:
            step_item_value, step_item_weight = self.internal_state[action_idx + 4][:]

            done = False

            if action_idx in self.actions_sequence:
                # DONE CHECK - #1
                done = True
                info['DoneReasonType'] = DoneReasonType0.TYPE_0   # "Selected Same items"
            else:
                self.actions_sequence.append(action_idx)

                # DONE CHECK - #2
                if self.weight_of_all_items_selected + step_item_weight <= self.config.LIMIT_WEIGHT_KNAPSACK:
                    self.items_selected.append(action_idx)
                    self.value_of_all_items_selected += step_item_value
                    self.weight_of_all_items_selected += step_item_weight

                    self.internal_state[2][0] += step_item_value
                    self.internal_state[2][1] += step_item_weight

                    # 해당 아이템 선택 표시
                    self.internal_state[action_idx + 4][:] = -1
                else:
                    done = True
                    info['DoneReasonType'] = DoneReasonType0.TYPE_1  # "Weight Limit Exceeded"

                if not done:
                    # DONE CHECK - #3
                    possible = False

                    for item in self.internal_state[4:]:
                        weight = item[1]

                        if weight != -1 and weight + self.internal_state[2][1] <= self.internal_state[0][1]:
                            possible = True
                            break

                    if not possible:
                        done = True
                        info['DoneReasonType'] = DoneReasonType0.TYPE_2  # "Weight Remains"

            if not done:
                self.num_step += 1
            self.internal_state[0][0] -= 1
        else:
            self.actions_sequence.append(action_idx)
            step_item_value, step_item_weight = self.internal_state[self.num_step + 4][:]

            done = False

            # DONE CHECK - #1
            if action_idx == 1:
                self.items_selected.append(self.num_step)

                if self.weight_of_all_items_selected + step_item_weight <= self.config.LIMIT_WEIGHT_KNAPSACK:
                    self.value_of_all_items_selected += step_item_value
                    self.weight_of_all_items_selected += step_item_weight

                    self.internal_state[2][0] += step_item_value
                    self.internal_state[2][1] += step_item_weight
                else:
                    done = True
                    info['DoneReasonType'] = DoneReasonType0.TYPE_1  # "Weight Limit Exceeded"

            if not done:
                # DONE CHECK - #2
                possible = False

                if self.num_step < self.config.NUM_ITEM - 1:
                    for item in self.internal_state[self.num_step + 5:]:
                        weight = item[1]

                        if weight + self.internal_state[2][1] <= self.internal_state[0][1]:
                            possible = True
                            break

                if self.num_step == self.config.NUM_ITEM - 1 or not possible:
                    done = True
                    info['DoneReasonType'] = DoneReasonType0.TYPE_2  # "Weight Remains"

            # LAST PROCESS
            self.internal_state[self.num_step + 4][:] = -1
            self.internal_state[0][0] -= 1

            if not done:
                self.num_step += 1
            self.internal_state[3][0] = self.internal_state[self.num_step + 4][0]
            self.internal_state[3][1] = self.internal_state[self.num_step + 4][1]

        if done:
            # TYPE_0 = "Selected Same items"
            # TYPE_1 = "Weight Limit Exceeded"
            if info['DoneReasonType'] in [DoneReasonType0.TYPE_0, DoneReasonType0.TYPE_1]:
                # self.last_ep_value_of_all_items_selected = 0.0
                # self.last_ep_weight_of_all_items_selected = 0.0
                # self.last_ep_ratio = 0.0
                self.last_ep_value_of_all_items_selected = self.value_of_all_items_selected
                self.last_ep_weight_of_all_items_selected = self.weight_of_all_items_selected

                self.last_ep_ratio = min(1.0, round(self.value_of_all_items_selected / self.optimal_value, 3))
            else:
                self.last_ep_value_of_all_items_selected = self.value_of_all_items_selected
                self.last_ep_weight_of_all_items_selected = self.weight_of_all_items_selected

                self.last_ep_ratio = min(1.0, round(self.value_of_all_items_selected / self.optimal_value, 3))

                if self.last_ep_solution_found[0] < self.value_of_all_items_selected:
                    self.last_ep_simple_solution_found = self.process_solution_found(self.last_ep_ratio)

                if self.config.USE_HER and self.current_goal <= self.value_of_all_items_selected:
                    self.current_goal = self.value_of_all_items_selected + 1
                    info['DoneReasonType'] = DoneReasonType0.TYPE_3

            reward = self.reward(done_type=info['DoneReasonType'])

        else:
            reward = self.reward(done_type=None)

        observation = self.observation()

        if self.config.USE_HER:
            self.internal_state[-1][0] = self.current_goal
            self.internal_state[-1][1] = self.current_goal
            info[HerConstant.ACHIEVED_GOAL] = self.last_ep_value_of_all_items_selected
            info[HerConstant.DESIRED_GOAL] = self.current_goal

            if done and info['DoneReasonType'] == DoneReasonType0.TYPE_2:  # "Weight Remains"
                info[HerConstant.HER_SAVE_DONE] = True
            else:
                info[HerConstant.HER_SAVE_DONE] = False

        info['Actions sequence'] = self.actions_sequence
        info['Items selected'] = self.items_selected
        info['internal_state'] = copy.deepcopy(self.internal_state)
        info['STRATEGY'] = self.config.STRATEGY

        info['last_ep_value_of_all_items_selected'] = self.last_ep_value_of_all_items_selected
        info['last_ep_weight_of_all_items_selected'] = self.last_ep_weight_of_all_items_selected
        info['last_ep_solution_found'] = self.last_ep_solution_found
        info['last_ep_simple_solution_found'] = self.last_ep_simple_solution_found
        info['last_ep_ratio'] = self.last_ep_ratio

        if done and self.config.PRINT_DETAILS_AT_EPISODE_END:
            self.print_knapsack_details_at_episode_end(info, self.last_ep_ratio)
            
        return observation, reward, done, info

    def process_solution_found(self, ratio):
        self.last_ep_solution_found[0] = self.value_of_all_items_selected
        self.last_ep_solution_found[1:] = self.items_selected

        self.last_ep_solution_found.append(ratio)

        simple_solution_found = [self.value_of_all_items_selected, ratio, self.episodes]

        if self.config.UPLOAD_PATH and self.config.INITIAL_ITEM_DISTRIBUTION_FIXED:
            upload_file('linklab', self.last_ep_solution_found, self.config.UPLOAD_PATH)

        return simple_solution_found

    def reward(self, done_type=None):
        if done_type is None:  # Normal Step
            mission_complete_reward = 0.0
            misbehavior_reward = 0.0
            value_of_all_items_selected_reward = 0.0
            goal_achieved_reward = 0.0

        elif done_type == DoneReasonType0.TYPE_0:  # "Selected Same Item"
            mission_complete_reward = 0.0
            misbehavior_reward = -1.0
            value_of_all_items_selected_reward = 0.0
            goal_achieved_reward = 0.0

        elif done_type == DoneReasonType0.TYPE_1:  # "Weight Limit Exceeded"
            mission_complete_reward = 0.0
            misbehavior_reward = -1.0
            value_of_all_items_selected_reward = 0.0
            goal_achieved_reward = 0.0

        elif done_type == DoneReasonType0.TYPE_2:  # "Weight Remains"
            mission_complete_reward = 0.0
            misbehavior_reward = 0.0
            value_of_all_items_selected_reward = self.value_of_all_items_selected / self.TOTAL_VALUE_FOR_ALL_ITEMS
            goal_achieved_reward = 0.0

        elif done_type == DoneReasonType0.TYPE_3:  # "Goal Achieved"
            mission_complete_reward = 0.0
            misbehavior_reward = 0.0
            value_of_all_items_selected_reward = 0.0
            goal_achieved_reward = 1.0

        else:
            raise ValueError()

        if self.config.USE_HER:
            reward = goal_achieved_reward
        else:
            reward = mission_complete_reward + misbehavior_reward + value_of_all_items_selected_reward

        return reward


def run_env():
    class Dummy_Agent:
        def get_action(self, observation):
            assert observation is not None
            available_action_ids = [0, 1]
            action_id = random.choice(available_action_ids)
            return action_id

    print("START RUN!!!")
    agent = Dummy_Agent()

    #Random Instance Test
    from link_rl.a_configuration.b_single_config.combinatorial_optimization.config_knapsack import \
        ConfigKnapsack0RandomTestLinearDqn
    config = ConfigKnapsack0RandomTestLinearDqn()

    #Load Instance Test
    from link_rl.a_configuration.b_single_config.combinatorial_optimization.config_knapsack import \
        ConfigKnapsack0LoadTestLinearDqn
    config = ConfigKnapsack0LoadTestLinearDqn()

    #Static Instance Test
    from link_rl.a_configuration.b_single_config.combinatorial_optimization.config_knapsack import \
        ConfigKnapsack0StaticTestDqn
    config = ConfigKnapsack0StaticTestDqn()


    set_config(config)

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


class KnapsackEnvStat(CustomEnvStat):
    def __init__(self):
        super(KnapsackEnvStat, self).__init__()
        self.test_items_value_lst = []
        self.test_ratio_value_lst = []

        self.test_last_avg_items_value = 0.0
        self.test_last_avg_ratio_value = 0.0

        self.train_last_ep_value_of_all_items_selected = 0.0
        self.train_last_ep_weight_of_all_items_selected = 0.0
        self.train_last_ep_simple_solution_found = None
        self.train_last_ep_ratio_value = 0.0

    def test_reset(self):
        self.test_items_value_lst.clear()
        self.test_ratio_value_lst.clear()

    def test_episode_done(self, info):
        self.test_items_value_lst.append(info["last_ep_value_of_all_items_selected"])
        self.test_ratio_value_lst.append(info["last_ep_ratio"])

    def test_evaluate(self):
        self.test_last_avg_items_value = np.average(self.test_items_value_lst)
        self.test_last_avg_ratio_value = np.average(self.test_ratio_value_lst)

    def test_evaluation_str(self):
        _test_evaluation_str = "Item value: {0:.2f}".format(self.test_last_avg_items_value)
        _test_evaluation_str += ", Ratio: {0:.2f}".format(self.test_last_avg_ratio_value)
        return _test_evaluation_str

    def train_evaluate(self, last_train_env_info):
        self.train_last_ep_value_of_all_items_selected = last_train_env_info["last_ep_value_of_all_items_selected"]
        self.train_last_ep_weight_of_all_items_selected = last_train_env_info["last_ep_weight_of_all_items_selected"]
        self.train_last_ep_simple_solution_found = last_train_env_info["last_ep_simple_solution_found"]
        if self.train_last_ep_simple_solution_found:
            self.train_last_ep_ratio_value = self.train_last_ep_simple_solution_found[1]

    def train_evaluation_str(self):
        _train_evaluation_str = "Items Value Selected: {0:5.1f}, Items Weight Selected: {1:5.1f}".format(
            self.train_last_ep_value_of_all_items_selected, self.train_last_ep_weight_of_all_items_selected
        )

        if self.train_last_ep_simple_solution_found is not None:
            _train_evaluation_str += ", Sol. Found: {0} (Ratio: {1:5.3f}, Ep. Found: {2:,})".format(
                self.train_last_ep_simple_solution_found[0],
                self.train_last_ep_simple_solution_found[1],
                self.train_last_ep_simple_solution_found[2]
            )

        return _train_evaluation_str

    def add_wandb_log(self, log_dict):
        log_dict["Value of All Item Selected"] = self.train_last_ep_value_of_all_items_selected
        log_dict["Ratio (Value to Optimal Value)"] = self.train_last_ep_ratio_value
        log_dict["[TEST] Value of All Item Selected"] = self.test_last_avg_items_value
        log_dict["[TEST] Ratio (Value to Optimal Value)"] = self.test_last_avg_ratio_value


class KnapsackEnvComparisonStat(CustomEnvComparisonStat):
    def __init__(self, n_runs, agents_labels, num_stat_data_size):
        super(KnapsackEnvComparisonStat, self).__init__(n_runs, agents_labels, num_stat_data_size)

        self.test_last_ep_value_of_all_items_selected_lst = []
        self.test_last_ep_ratio_lst = []

        self.test_last_avg_ep_value_of_all_items_selected = 0.0
        self.test_last_avg_ep_ratio = 0.0

        self.train_last_ep_value_of_all_items_selected_per_agent = []
        self.train_last_ep_ratio_per_agent = []

        self.test_value_of_items_selected_per_agent = np.zeros((n_runs, self.n_agents, self.num_stat_data_size))
        self.MIN_test_value_of_items_selected_per_agent = np.zeros((self.n_agents, self.num_stat_data_size))
        self.MEAN_test_value_of_items_selected_per_agent = np.zeros((self.n_agents, self.num_stat_data_size))
        self.MAX_test_value_of_items_selected_per_agent = np.zeros((self.n_agents, self.num_stat_data_size))

        self.train_value_of_items_selected_per_agent = np.zeros((n_runs, self.n_agents, self.num_stat_data_size))
        self.MIN_train_value_of_items_selected_per_agent = np.zeros((self.n_agents, self.num_stat_data_size))
        self.MEAN_train_value_of_items_selected_per_agent = np.zeros((self.n_agents, self.num_stat_data_size))
        self.MAX_train_value_of_items_selected_per_agent = np.zeros((self.n_agents, self.num_stat_data_size))

        self.test_ratio_value_to_optimal_value = np.zeros((n_runs, self.n_agents, self.num_stat_data_size))
        self.MIN_test_ratio_value_to_optimal_value = np.zeros((self.n_agents, self.num_stat_data_size))
        self.MEAN_test_ratio_value_to_optimal_value = np.zeros((self.n_agents, self.num_stat_data_size))
        self.MAX_test_ratio_value_to_optimal_value = np.zeros((self.n_agents, self.num_stat_data_size))

        self.train_ratio_value_to_optimal_value = np.zeros((n_runs, self.n_agents, self.num_stat_data_size))
        self.MIN_train_ratio_value_to_optimal_value = np.zeros((self.n_agents, self.num_stat_data_size))
        self.MEAN_train_ratio_value_to_optimal_value = np.zeros((self.n_agents, self.num_stat_data_size))
        self.MAX_train_ratio_value_to_optimal_value = np.zeros((self.n_agents, self.num_stat_data_size))

    def test_reset(self):
        self.test_last_ep_value_of_all_items_selected_lst.clear()
        self.test_last_ep_ratio_lst.clear()

    def test_episode_done(self, info):
        self.test_last_ep_value_of_all_items_selected_lst.append(info['last_ep_value_of_all_items_selected'])
        self.test_last_ep_ratio_lst.append(info['last_ep_ratio'])

    def test_evaluate(self):
        self.test_last_avg_ep_value_of_all_items_selected = np.average(self.test_last_ep_value_of_all_items_selected_lst)
        self.test_last_avg_ep_ratio = np.average(self.test_last_ep_ratio_lst)

    def test_evaluation_str(self):
        _test_evaliation_str = "Items Value Selected - Average: {0:5.1f}, Ratio (Value to Optimal Value): {1:5.1f})".format(
                self.test_last_avg_ep_value_of_all_items_selected, self.test_last_avg_ep_ratio
        )
        return _test_evaliation_str

    def train_reset(self):
        self.train_last_ep_value_of_all_items_selected_per_agent.append(0.0)
        self.train_last_ep_ratio_per_agent.append(0.0)

    def train_evaluate(self, agent_idx, last_train_env_info):
        self.train_last_ep_value_of_all_items_selected_per_agent[agent_idx] = \
            last_train_env_info['last_ep_weight_of_all_items_selected']
        self.train_last_ep_ratio_per_agent[agent_idx] = last_train_env_info['last_ep_ratio']

    def update_stat(self, run, agent_idx, stat_idx, target_stats, min_target_stats, max_target_stats, mean_target_stats):
        self.test_value_of_items_selected_per_agent[run, agent_idx, stat_idx] = \
            self.test_last_avg_ep_value_of_all_items_selected

        self.train_value_of_items_selected_per_agent[run, agent_idx, stat_idx] = \
            self.train_last_ep_value_of_all_items_selected_per_agent[agent_idx]

        self.test_ratio_value_to_optimal_value[run, agent_idx, stat_idx] = \
            self.test_last_avg_ep_ratio

        self.train_ratio_value_to_optimal_value[run, agent_idx, stat_idx] = \
            self.train_last_ep_ratio_per_agent[agent_idx]

        target_stats.append(self.test_value_of_items_selected_per_agent)
        target_stats.append(self.train_value_of_items_selected_per_agent)
        target_stats.append(self.test_ratio_value_to_optimal_value)
        target_stats.append(self.train_ratio_value_to_optimal_value)

        min_target_stats.append(self.MIN_test_value_of_items_selected_per_agent)
        min_target_stats.append(self.MIN_train_value_of_items_selected_per_agent)
        min_target_stats.append(self.MIN_test_ratio_value_to_optimal_value)
        min_target_stats.append(self.MIN_train_ratio_value_to_optimal_value)

        max_target_stats.append(self.MAX_test_value_of_items_selected_per_agent)
        max_target_stats.append(self.MAX_train_value_of_items_selected_per_agent)
        max_target_stats.append(self.MAX_test_ratio_value_to_optimal_value)
        max_target_stats.append(self.MAX_train_ratio_value_to_optimal_value)

        mean_target_stats.append(self.MEAN_test_value_of_items_selected_per_agent)
        mean_target_stats.append(self.MEAN_train_value_of_items_selected_per_agent)
        mean_target_stats.append(self.MEAN_test_ratio_value_to_optimal_value)
        mean_target_stats.append(self.MEAN_train_ratio_value_to_optimal_value)

    def add_wandb_log_comparison(self, plotly_layout, training_steps_str, run, test_training_steps_lst, log_dict):
        ###############################################################################
        plotly_layout.yaxis.title = "[TEST] Value of Items Selected"
        plotly_layout.xaxis.title = "Training Steps ({0}, runs={1}, over {2} Episodes)".format(
            training_steps_str, run + 1, self.n_episodes_for_mean_calculation
        )
        data = []
        for agent_idx in range(self.n_agents):
            data.append(
                go.Scatter(
                    name=self.agent_labels[agent_idx],
                    x=test_training_steps_lst,
                    y=self.MEAN_test_value_of_items_selected_per_agent[agent_idx, :],
                    showlegend=True
                )
            )

        test_value_of_items_selected_per_agent = go.Figure(data=data, layout=plotly_layout)

        ###############################################################################
        plotly_layout.yaxis.title = "[TEST] Ratio (Value to Optimal Value)"
        plotly_layout.xaxis.title = "Training Steps ({0}, runs={1}, over {2} Episodes)".format(
            training_steps_str, run + 1, self.n_episodes_for_mean_calculation
        )
        data = []
        for agent_idx in range(self.n_agents):
            data.append(
                go.Scatter(
                    name=self.agent_labels[agent_idx],
                    x=test_training_steps_lst,
                    y=self.MEAN_test_ratio_value_to_optimal_value[agent_idx, :],
                    showlegend=True
                )
            )

        test_ratio_value_to_optimal_value = go.Figure(data=data, layout=plotly_layout)

        ###############################################################################
        plotly_layout.yaxis.title = "[TRAIN] Value of Items Selected"
        plotly_layout.xaxis.title = "Training Steps ({0}, runs={1}, over {2} Episodes)".format(
            training_steps_str, run + 1, self.n_episodes_for_mean_calculation
        )
        data = []
        for agent_idx in range(self.n_agents):
            data.append(
                go.Scatter(
                    name=self.agent_labels[agent_idx],
                    x=test_training_steps_lst,
                    y=self.MEAN_train_value_of_items_selected_per_agent[agent_idx, :],
                    showlegend=True
                )
            )

        train_value_of_items_selected_per_agent = go.Figure(data=data, layout=plotly_layout)

        ###############################################################################
        plotly_layout.yaxis.title = "[TRAIN] Ratio (Value to Optimal Value)"
        plotly_layout.xaxis.title = "Training Steps ({0}, runs={1})".format(
            training_steps_str, run + 1
        )
        data = []
        for agent_idx in range(self.n_agents):
            data.append(
                go.Scatter(
                    name=self.agent_labels[agent_idx],
                    x=test_training_steps_lst,
                    y=self.MEAN_train_ratio_value_to_optimal_value[agent_idx, :],
                    showlegend=True
                )
            )

        train_ratio_value_to_optimal_value = go.Figure(data=data, layout=plotly_layout)

        log_dict["[TEST] value_of_items_selected_per_agent"] = test_value_of_items_selected_per_agent
        log_dict["[TEST] ratio_value_to_optimal_value"] = test_ratio_value_to_optimal_value
        log_dict["[TRAIN] value_of_items_selected_per_agent"] = train_value_of_items_selected_per_agent
        log_dict["[TRAIN] ratio_value_to_optimal_value"] = train_ratio_value_to_optimal_value


if __name__ == "__main__":
    run_env()
