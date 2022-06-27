import gym
from gym import spaces
import enum
import numpy as np
import copy
from typing import Optional
import random
import datetime as dt

from a_configuration.a_base_config.a_environments.combinatorial_optimization.config_knapsack import ConfigKnapsack0, \
    ConfigKnapsackTest, ConfigKnapsackStaticTest
from a_configuration.a_base_config.c_models.config_convolutional_models import Config1DConvolutionalModel, \
    Config2DConvolutionalModel
from a_configuration.a_base_config.c_models.config_linear_models import ConfigLinearModel
from a_configuration.a_base_config.config_parse import SYSTEM_USER_NAME, SYSTEM_COMPUTER_NAME
from a_configuration.b_single_config.combinatorial_optimization.config_knapsack import ConfigKnapsackStaticTestDqn
from b_environments.combinatorial_optimization.boto3_knapsack import load_instance, upload_file, load_solution
from b_environments.combinatorial_optimization.knapsack_gurobi import model_kp
from g_utils.types import ModelType

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

        self.TOTAL_VALUE_FOR_ALL_ITEMS = None

        self.INITIAL_ITEM_DISTRIBUTION_FIXED = config.INITIAL_ITEM_DISTRIBUTION_FIXED

        self.FILE_PATH = config.FILE_PATH
        self.UPLOAD_PATH = config.UPLOAD_PATH
        self.OPTIMAL_PATH = config.OPTIMAL_PATH
        self.INSTANCE_INDEX = config.INSTANCE_INDEX
        self.SORTING_TYPE = config.SORTING_TYPE

        self.solution_found = config.SOLUTION_FOUND
        self.optimal_value = 0

        self.internal_state = None
        self.items_selected = None
        self.actions_sequence = None
        self.weight_of_all_items_selected = None
        self.value_of_all_items_selected = None
        self.num_step = None

        self.action_space = spaces.Discrete(2)

        if isinstance(config.MODEL_PARAMETER, ConfigLinearModel):
            self.observation_space = spaces.Box(
                low=-1.0, high=1000.0,
                shape=(2*(self.NUM_ITEM + 4),)
            )
        elif isinstance(config.MODEL_PARAMETER, Config2DConvolutionalModel):
            self.observation_space = spaces.Box(
                low=-1.0, high=1000.0,
                shape=(1, self.NUM_ITEM + 2, 4)
            )
        elif isinstance(config.MODEL_PARAMETER, Config1DConvolutionalModel):
            self.observation_space = spaces.Box(
                low=-1.0, high=1000.0,
                shape=(self.NUM_ITEM + 2, 4)
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
        else:
            state = np.zeros(shape=(self.NUM_ITEM + 4, 2), dtype=float)

            if self.FILE_PATH:
                self.FILE_PATH = self.FILE_PATH + '/instance' + str(self.INSTANCE_INDEX) + '.csv'
                data = load_instance('linklab', self.FILE_PATH)

                state = data

                for item_idx in range(self.NUM_ITEM):
                    state[1][0] += state[item_idx+4][0]
                    state[1][1] += state[item_idx+4][1]

                self.LIMIT_WEIGHT_KNAPSACK = state[0][1]
                state[0][0] = self.NUM_ITEM

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

                    state[self.NUM_ITEM][2] += item_value
                    state[self.NUM_ITEM][3] += item_weight

                    state[-1][2] += item_value
                    state[-1][3] += item_weight

                state[-1][1] = np.array(self.LIMIT_WEIGHT_KNAPSACK)

            if self.UPLOAD_PATH:
                date = dt.datetime.now()
                date_str = '/' + str(date.year) + str(date.month) + str(date.day)
                user = SYSTEM_USER_NAME
                com = SYSTEM_COMPUTER_NAME
                self.UPLOAD_PATH = self.UPLOAD_PATH + date_str + user + com + '/link_solution' + str(self.INSTANCE_INDEX) + '.csv'
            else:
                self.UPLOAD_PATH = 'knapsack_instances/TEST/link_solution'
                date = dt.datetime.now()
                date_str = '/' + str(date.year) + str(date.month) + str(date.day)
                user = SYSTEM_USER_NAME
                com = SYSTEM_COMPUTER_NAME
                self.UPLOAD_PATH = self.UPLOAD_PATH + date_str + user + com + '/link_solution' + str(self.INSTANCE_INDEX) + '.csv'

            if self.OPTIMAL_PATH:
                self.OPTIMAL_PATH = self.OPTIMAL_PATH + '/solution' + str(self.INSTANCE_INDEX) + '.csv'
                self.optimal_value = load_solution('linklab', self.OPTIMAL_PATH)
            else:
                Knapsack_capacity = float(state[-1][1])
                values = state[:-1, 2]
                weights = state[:-1, 3]

                items_selected, self.optimal_value = model_kp(Knapsack_capacity, values, weights, False)

                self.OPTIMAL_PATH = 'knapsack_instances/TEST/optimal_solution'
                date = dt.datetime.now()
                date_str = '/' + str(date.year) + str(date.month) + str(date.day)
                user = SYSTEM_USER_NAME
                com = SYSTEM_COMPUTER_NAME
                self.OPTIMAL_PATH = self.OPTIMAL_PATH + date_str + user + com + '/optimal_solution' + str(self.INSTANCE_INDEX) + '.csv'

                upload_file('linklab', (items_selected, self.optimal_value), self.OPTIMAL_PATH)

        return state

    def state_sorting(self, state, start, end):

        if self.SORTING_TYPE == 1: #Value Per Weight
            if start >= end:
                return

            pivot = start
            left = start + 1
            right = end

            while left <= right:
                while left <= end and (state[left][2] / state[left][3]) <= (state[pivot][2] / state[pivot][3]):
                    left += 1

                while right > start and (state[right][2] / state[right][3]) >= (state[pivot][2] / state[pivot][3]):
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
                while left <= end and state[left][2] <= state[pivot][2]:
                    left += 1

                while right > start and state[right][2] >= state[pivot][2]:
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
                while left <= end and state[left][3] <= state[pivot][3]:
                    left += 1

                while right > start and state[right][3] >= state[pivot][3]:
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
        elif isinstance(self.config.MODEL_PARAMETER, Config2DConvolutionalModel):
            observation = copy.deepcopy(self.internal_state) / self.LIMIT_WEIGHT_KNAPSACK
        else:
            raise ValueError()
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

        self.TOTAL_VALUE_FOR_ALL_ITEMS = sum(self.internal_state[:, 0])
        self.items_selected = []
        self.actions_sequence = []
        self.weight_of_all_items_selected = 0
        self.value_of_all_items_selected = 0

        self.num_step = 0
        self.internal_state[3][0] = self.internal_state[4][0]
        self.internal_state[3][1] = self.internal_state[4][1]

        observation = self.observation()
        info = dict()
        info['internal_state'] = copy.deepcopy(self.internal_state)

        if not self.FILE_PATH:
            self.FILE_PATH = 'knapsack_instances/TEST/instances'
            date = dt.datetime.now()
            date_str = '/' + str(date.year) + str(date.month) + str(date.day)
            user = SYSTEM_USER_NAME
            com = SYSTEM_COMPUTER_NAME
            self.FILE_PATH = self.FILE_PATH + date_str + user + com + '/instance' + str(
                self.INSTANCE_INDEX) + '.csv'

            upload_file('linklab', self.internal_state, self.FILE_PATH)


        if return_info:
            return observation[None, :], info
        else:
            return observation[None, :]

    def check_future_select_possible(self):
        possible = False

        for item in self.internal_state[self.num_step + 5:-1]:
            weight = item[1]

            if weight + self.internal_state[2][1] <= self.internal_state[0][1]:
                possible = True
                break

        return possible

    def step(self, action_idx):
        self.actions_sequence.append(action_idx)
        info = dict()
        step_item_value, step_item_weight = self.internal_state[self.num_step+4][:]

        if action_idx == 1:
            self.items_selected.append(self.num_step)

            self.value_of_all_items_selected += step_item_value
            self.weight_of_all_items_selected += step_item_weight

            self.internal_state[2][0] += step_item_value
            self.internal_state[2][1] += step_item_weight

        possible = self.check_future_select_possible()

        done = False

        self.internal_state[self.num_step+4][:] = -1

        if self.num_step == self.NUM_ITEM - 1 or not possible:
            done = True

            if self.weight_of_all_items_selected > self.LIMIT_WEIGHT_KNAPSACK:
                info['DoneReasonType'] = DoneReasonType0.TYPE_1  # "Weight Limit Exceeded"
            else:
                info['DoneReasonType'] = DoneReasonType0.TYPE_2  # "Weight Remains"

            if self.num_step != self.NUM_ITEM - 1:
                self.num_step += 1
                self.internal_state[3][0] = self.internal_state[self.num_step + 4][0]
                self.internal_state[3][1] = self.internal_state[self.num_step + 4][1]

        else:
            self.num_step += 1
            self.internal_state[3][0] = self.internal_state[self.num_step + 4][0]
            self.internal_state[3][1] = self.internal_state[self.num_step + 4][1]

        self.internal_state[0][0] -= 1
        observation = self.observation()

        if done:
            reward = self.reward(done_type=info['DoneReasonType'])

            if info['DoneReasonType'] != DoneReasonType0.TYPE_1:
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
            available_action_ids = [0, 1]
            action_id = random.choice(available_action_ids)
            return action_id

    print("START RUN!!!")
    agent = Dummy_Agent()

    config = ConfigKnapsack0()
    config.NUM_ITEM = 50
    if config.MODEL_TYPE in (
        ModelType.TINY_1D_CONVOLUTIONAL, ModelType.SMALL_1D_CONVOLUTIONAL,
        ModelType.MEDIUM_1D_CONVOLUTIONAL, ModelType.LARGE_1D_CONVOLUTIONAL
    ):
        from a_configuration.a_base_config.c_models.config_convolutional_models import Config1DConvolutionalModel
        config.MODEL_PARAMETER = Config1DConvolutionalModel(config.MODEL_TYPE)

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
import gym
from gym import spaces
import enum
import numpy as np
import copy
from typing import Optional
import random
import datetime as dt

from a_configuration.a_base_config.a_environments.combinatorial_optimization.config_knapsack import ConfigKnapsack0, \
    ConfigKnapsackTest, ConfigKnapsackStaticTest
from a_configuration.a_base_config.c_models.config_convolutional_models import Config1DConvolutionalModel, \
    Config2DConvolutionalModel
from a_configuration.a_base_config.c_models.config_linear_models import ConfigLinearModel
from a_configuration.a_base_config.config_parse import SYSTEM_USER_NAME, SYSTEM_COMPUTER_NAME
from a_configuration.b_single_config.combinatorial_optimization.config_knapsack import ConfigKnapsackStaticTestDqn
from b_environments.combinatorial_optimization.boto3_knapsack import load_instance, upload_file, load_solution
from b_environments.combinatorial_optimization.knapsack_gurobi import model_kp
from g_utils.types import ModelType

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

        self.TOTAL_VALUE_FOR_ALL_ITEMS = None

        self.INITIAL_ITEM_DISTRIBUTION_FIXED = config.INITIAL_ITEM_DISTRIBUTION_FIXED

        self.FILE_PATH = config.FILE_PATH
        self.UPLOAD_PATH = config.UPLOAD_PATH
        self.OPTIMAL_PATH = config.OPTIMAL_PATH
        self.INSTANCE_INDEX = config.INSTANCE_INDEX
        self.SORTING_TYPE = config.SORTING_TYPE

        self.solution_found = config.SOLUTION_FOUND
        self.optimal_value = 0

        self.internal_state = None
        self.items_selected = None
        self.actions_sequence = None
        self.weight_of_all_items_selected = None
        self.value_of_all_items_selected = None
        self.num_step = None

        self.action_space = spaces.Discrete(2)

        if isinstance(config.MODEL_PARAMETER, ConfigLinearModel):
            self.observation_space = spaces.Box(
                low=-1.0, high=1000.0,
                shape=(2*(self.NUM_ITEM + 4),)
            )
        elif isinstance(config.MODEL_PARAMETER, Config2DConvolutionalModel):
            self.observation_space = spaces.Box(
                low=-1.0, high=1000.0,
                shape=(1, self.NUM_ITEM + 2, 4)
            )
        elif isinstance(config.MODEL_PARAMETER, Config1DConvolutionalModel):
            self.observation_space = spaces.Box(
                low=-1.0, high=1000.0,
                shape=(self.NUM_ITEM + 2, 4)
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
        else:
            state = np.zeros(shape=(self.NUM_ITEM + 4, 2), dtype=float)

            if self.FILE_PATH:
                self.FILE_PATH = self.FILE_PATH + '/instance' + str(self.INSTANCE_INDEX) + '.csv'
                data = load_instance('linklab', self.FILE_PATH)

                state = data

                for item_idx in range(self.NUM_ITEM):
                    state[1][0] += state[item_idx+4][0]
                    state[1][1] += state[item_idx+4][1]

                self.LIMIT_WEIGHT_KNAPSACK = state[0][1]
                state[0][0] = self.NUM_ITEM

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

                    state[self.NUM_ITEM][2] += item_value
                    state[self.NUM_ITEM][3] += item_weight

                    state[-1][2] += item_value
                    state[-1][3] += item_weight

                state[-1][1] = np.array(self.LIMIT_WEIGHT_KNAPSACK)

            if self.UPLOAD_PATH:
                date = dt.datetime.now()
                date_str = '/' + str(date.year) + str(date.month) + str(date.day)
                user = SYSTEM_USER_NAME
                com = SYSTEM_COMPUTER_NAME
                self.UPLOAD_PATH = self.UPLOAD_PATH + date_str + user + com + '/link_solution' + str(self.INSTANCE_INDEX) + '.csv'
            else:
                self.UPLOAD_PATH = 'knapsack_instances/TEST/link_solution'
                date = dt.datetime.now()
                date_str = '/' + str(date.year) + str(date.month) + str(date.day)
                user = SYSTEM_USER_NAME
                com = SYSTEM_COMPUTER_NAME
                self.UPLOAD_PATH = self.UPLOAD_PATH + date_str + user + com + '/link_solution' + str(self.INSTANCE_INDEX) + '.csv'

            if self.OPTIMAL_PATH:
                self.OPTIMAL_PATH = self.OPTIMAL_PATH + '/solution' + str(self.INSTANCE_INDEX) + '.csv'
                self.optimal_value = load_solution('linklab', self.OPTIMAL_PATH)
            else:
                Knapsack_capacity = float(state[-1][1])
                values = state[:-1, 2]
                weights = state[:-1, 3]

                items_selected, self.optimal_value = model_kp(Knapsack_capacity, values, weights, False)

                self.OPTIMAL_PATH = 'knapsack_instances/TEST/optimal_solution'
                date = dt.datetime.now()
                date_str = '/' + str(date.year) + str(date.month) + str(date.day)
                user = SYSTEM_USER_NAME
                com = SYSTEM_COMPUTER_NAME
                self.OPTIMAL_PATH = self.OPTIMAL_PATH + date_str + user + com + '/optimal_solution' + str(self.INSTANCE_INDEX) + '.csv'

                upload_file('linklab', (items_selected, self.optimal_value), self.OPTIMAL_PATH)

        return state

    def state_sorting(self, state, start, end):

        if self.SORTING_TYPE == 1: #Value Per Weight
            if start >= end:
                return

            pivot = start
            left = start + 1
            right = end

            while left <= right:
                while left <= end and (state[left][2] / state[left][3]) <= (state[pivot][2] / state[pivot][3]):
                    left += 1

                while right > start and (state[right][2] / state[right][3]) >= (state[pivot][2] / state[pivot][3]):
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
                while left <= end and state[left][2] <= state[pivot][2]:
                    left += 1

                while right > start and state[right][2] >= state[pivot][2]:
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
                while left <= end and state[left][3] <= state[pivot][3]:
                    left += 1

                while right > start and state[right][3] >= state[pivot][3]:
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
        elif isinstance(self.config.MODEL_PARAMETER, Config2DConvolutionalModel):
            observation = copy.deepcopy(self.internal_state) / self.LIMIT_WEIGHT_KNAPSACK
        else:
            raise ValueError()
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

        self.TOTAL_VALUE_FOR_ALL_ITEMS = sum(self.internal_state[:, 0])
        self.items_selected = []
        self.actions_sequence = []
        self.weight_of_all_items_selected = 0
        self.value_of_all_items_selected = 0

        self.num_step = 0
        self.internal_state[3][0] = self.internal_state[4][0]
        self.internal_state[3][1] = self.internal_state[4][1]

        observation = self.observation()
        info = dict()
        info['internal_state'] = copy.deepcopy(self.internal_state)

        if not self.FILE_PATH:
            self.FILE_PATH = 'knapsack_instances/TEST/instances'
            date = dt.datetime.now()
            date_str = '/' + str(date.year) + str(date.month) + str(date.day)
            user = SYSTEM_USER_NAME
            com = SYSTEM_COMPUTER_NAME
            self.FILE_PATH = self.FILE_PATH + date_str + user + com + '/instance' + str(
                self.INSTANCE_INDEX) + '.csv'

            upload_file('linklab', self.internal_state, self.FILE_PATH)


        if return_info:
            return observation[None, :], info
        else:
            return observation[None, :]

    def check_future_select_possible(self):
        possible = False

        for item in self.internal_state[self.num_step + 5:-1]:
            weight = item[1]

            if weight + self.internal_state[2][1] <= self.internal_state[0][1]:
                possible = True
                break

        return possible

    def step(self, action_idx):
        self.actions_sequence.append(action_idx)
        info = dict()
        step_item_value, step_item_weight = self.internal_state[self.num_step+4][:]

        if action_idx == 1:
            self.items_selected.append(self.num_step)

            self.value_of_all_items_selected += step_item_value
            self.weight_of_all_items_selected += step_item_weight

            self.internal_state[2][0] += step_item_value
            self.internal_state[2][1] += step_item_weight

        possible = self.check_future_select_possible()

        done = False

        self.internal_state[self.num_step+4][:] = -1

        if self.num_step == self.NUM_ITEM - 1 or not possible:
            done = True

            if self.weight_of_all_items_selected > self.LIMIT_WEIGHT_KNAPSACK:
                info['DoneReasonType'] = DoneReasonType0.TYPE_1  # "Weight Limit Exceeded"
            else:
                info['DoneReasonType'] = DoneReasonType0.TYPE_2  # "Weight Remains"

            if self.num_step != self.NUM_ITEM - 1:
                self.num_step += 1
                self.internal_state[3][0] = self.internal_state[self.num_step + 4][0]
                self.internal_state[3][1] = self.internal_state[self.num_step + 4][1]

        else:
            self.num_step += 1
            self.internal_state[3][0] = self.internal_state[self.num_step + 4][0]
            self.internal_state[3][1] = self.internal_state[self.num_step + 4][1]

        self.internal_state[0][0] -= 1
        observation = self.observation()

        if done:
            reward = self.reward(done_type=info['DoneReasonType'])

            if info['DoneReasonType'] != DoneReasonType0.TYPE_1:
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
            available_action_ids = [0, 1]
            action_id = random.choice(available_action_ids)
            return action_id

    print("START RUN!!!")
    agent = Dummy_Agent()

    config = ConfigKnapsack0()
    config.NUM_ITEM = 50
    if config.MODEL_TYPE in (
        ModelType.TINY_1D_CONVOLUTIONAL, ModelType.SMALL_1D_CONVOLUTIONAL,
        ModelType.MEDIUM_1D_CONVOLUTIONAL, ModelType.LARGE_1D_CONVOLUTIONAL
    ):
        from a_configuration.a_base_config.c_models.config_convolutional_models import Config1DConvolutionalModel
        config.MODEL_PARAMETER = Config1DConvolutionalModel(config.MODEL_TYPE)

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
