from collections import deque

from link_rl.a_configuration.a_base_config.c_models.config_convolutional_models import Config1DConvolutionalModel
from link_rl.a_configuration.a_base_config.c_models.config_linear_models import ConfigLinearModel
from link_rl.a_configuration.a_base_config.c_models.config_recurrent_convolutional_models import ConfigRecurrent1DConvolutionalModel
from link_rl.a_configuration.a_base_config.c_models.config_recurrent_linear_models import ConfigRecurrentLinearModel
from link_rl.g_utils.types import Transition



class HerEpisodeBuffer:
    def __init__(self, observation_space, action_space, config):
        self.observation_space = observation_space
        self.action_space = action_space
        self.episode_buffer = None
        self.config = config

    def reset(self):
        self.episode_buffer = deque()

    def append(self, transition):
        self.episode_buffer.append(transition)

    def size(self):
        return len(self.episode_buffer)

    def _get_observation_and_goal(self, observation, her_goal):
        if self.config.ENV_NAME in ["Knapsack_Problem_v0"]:
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
        new_episode_trajectory = deque()

        for idx, transition in enumerate(self.episode_buffer):
            new_episode_trajectory.append(Transition(
                observation=self._get_observation_and_goal(transition.observation, her_goal),
                action=transition.action,
                next_observation=self._get_observation_and_goal(transition.next_observation, her_goal),
                reward=1.0 if idx == len(self.episode_buffer) - 1 else 0.0,
                done=True if idx == len(self.episode_buffer) - 1 else False,
                info=transition.info
            ))

        #print(new_episode_buffer, "!!!")
        return new_episode_trajectory
