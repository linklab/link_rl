from collections import deque

from b_environments.combinatorial_optimization.knapsack.knapsack import *
from g_utils.types import HerConstant, Transition


class HerEpisodeBuffer:
    def __init__(self, config):
        self.episode_buffer = None
        self.config = config

    def reset(self):
        self.episode_buffer = deque()

    def save(self, transition):
        self.episode_buffer.append(transition)

    def size(self):
        return len(self.episode_buffer)

    def _get_observation_and_goal(self, observation, her_goal):
        if self.config.ENV_NAME in ["Her_Knapsack_Problem_v0"]:
            normalized_her_goal = her_goal / self.config.LIMIT_WEIGHT_KNAPSACK
            if isinstance(self.config.MODEL_PARAMETER, (ConfigLinearModel, ConfigRecurrentLinearModel)):
                observation[-1] = normalized_her_goal
                observation[-2] = normalized_her_goal
            elif isinstance(self.config.MODEL_PARAMETER, (Config1DConvolutionalModel, ConfigRecurrent1DConvolutionalModel)):
                observation[-1][0] = normalized_her_goal
                observation[-1][1] = normalized_her_goal
            else:
                raise ValueError()

            return observation
        else:
            raise ValueError()

    def get_her_trajectory(self, her_goal):
        new_episode_buffer = deque()

        for idx, transition in enumerate(self.episode_buffer):
            new_episode_buffer.append(Transition(
                observation=self._get_observation_and_goal(transition.observation, her_goal),
                action=transition.action,
                next_observation=self._get_observation_and_goal(transition.next_observation, her_goal),
                reward=1.0 if idx == len(self.episode_buffer) - 1 else 0.0,
                done=True if idx == len(self.episode_buffer) - 1 else False,
                info=transition.info
            ))

        #print(new_episode_buffer, "!!!")
        return new_episode_buffer


class HerKnapsackEnv(KnapsackEnv):
    def __init__(self, config):
        super().__init__(config)

        if isinstance(config.MODEL_PARAMETER, (ConfigLinearModel, ConfigRecurrentLinearModel)):
            self.observation_space = spaces.Box(
                low=-1.0, high=1000.0,
                shape=((self.NUM_ITEM + 5) * 2,)
            )
        elif isinstance(config.MODEL_PARAMETER, (Config1DConvolutionalModel, ConfigRecurrent1DConvolutionalModel)):
            self.observation_space = spaces.Box(
                low=-1.0, high=1000.0,
                shape=(self.NUM_ITEM + 5, 2)
            )
        else:
            raise ValueError()

        self.current_goal = 0

    def reset(self, *, seed: Optional[int] = None, return_info: bool = False, options: Optional[dict] = None,):
        observation, info = super().reset(return_info=return_info)
        info[HerConstant.ACHIEVED_GOAL] = self.current_goal
        info[HerConstant.DESIRED_GOAL] = self.current_goal

        goal_array = np.asarray([self.current_goal, self.current_goal])
        self.internal_state = np.vstack([self.internal_state, goal_array])
        info['internal_state'] = copy.deepcopy(self.internal_state)

        observation = self.observation()

        if return_info:
            return observation, info
        else:
            return observation

    def step(self, action_idx):
        observation, reward, done, info = super().step(action_idx=action_idx)

        if self.current_goal < self.value_of_all_items_selected:
            self.current_goal = self.value_of_all_items_selected + 1
            info['DoneReasonType'] = DoneReasonType0.TYPE_4
            done = True

        info[HerConstant.ACHIEVED_GOAL] = self.value_of_all_items_selected
        info[HerConstant.DESIRED_GOAL] = self.current_goal

        if done:
            reward = self.reward(done_type=info['DoneReasonType'])

            if info['DoneReasonType'] != DoneReasonType0.TYPE_1:  # "Weight Limit Exceeded"
                if self.solution_found[0] < self.value_of_all_items_selected:
                    self.process_solution_found()

            if info['DoneReasonType'] == DoneReasonType0.TYPE_2:
                info[HerConstant.HER_SAVE_DONE] = True
            else:
                info[HerConstant.HER_SAVE_DONE] = False
        else:
            reward = self.reward(done_type=None)

        self.internal_state[-1][0] = self.current_goal
        self.internal_state[-1][1] = self.current_goal
        info['internal_state'] = copy.deepcopy(self.internal_state)

        observation = self.observation()

        return observation, reward, done, info

    def reward(self, done_type=None):
        if done_type is None:  # Normal Step
            goal_achieved = 0.0
            mission_complete_reward = 0.0
            misbehavior_reward = 0.0

        elif done_type == DoneReasonType0.TYPE_1:  # "Weight Limit Exceeded"
            goal_achieved = 0.0
            mission_complete_reward = 0.0
            misbehavior_reward = -1.0

        elif done_type == DoneReasonType0.TYPE_2:  # "Weight Remains"
            goal_achieved = 0.0
            mission_complete_reward = 0.0
            misbehavior_reward = 0.0

        elif done_type == DoneReasonType0.TYPE_3:  # "All Item Selected"
            goal_achieved = 1.0
            mission_complete_reward = 1.0
            misbehavior_reward = 0.0

        elif done_type == DoneReasonType0.TYPE_4:  # "Goal Achieved"
            goal_achieved = 1.0
            mission_complete_reward = 0.0
            misbehavior_reward = 0.0

        else:
            raise ValueError()

        return goal_achieved + mission_complete_reward + misbehavior_reward


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
    from a_configuration.b_single_config.combinatorial_optimization.config_her_knapsack import \
        ConfigHerKnapsack0StaticTestLinearDoubleDqn
    config = ConfigHerKnapsack0StaticTestLinearDoubleDqn()
    config.STATIC_INITIAL_STATE_50 = False
    config.SORTING_TYPE = 1
    config.NUM_ITEM = 5
    set_config(config)

    env = HerKnapsackEnv(config)

    for i in range(3):
        observation, info = env.reset(return_info=True)
        done = False
        print("EPISODE: {0} ".format(i + 1) + "#" * 50)
        while not done:
            action = agent.get_action(observation)
            next_observation, reward, done, next_info = env.step(action)

            print("Observation: \n{0}, \nAction: {1}, next_observation: \n{2}, Reward: {3}, Done: {4}, "
                  "Achieved Goal: {5}, Desired Goal: {6}, ".format(
                info['internal_state'], action, next_info['internal_state'], reward, done,
                next_info[HerConstant.ACHIEVED_GOAL], next_info[HerConstant.DESIRED_GOAL]
            ), end="")
            if done:
                print("({0}: {1}, HER SAVE: {2})\n".format(
                    next_info['DoneReasonType'], next_info['DoneReasonType'].value, next_info[HerConstant.HER_SAVE_DONE]
                ))
            else:
                print("\n")
            observation = next_observation
            info = next_info


if __name__ == "__main__":
    run_env()