from collections import deque

import torch
from gym.spaces import Discrete

from g_utils.buffers.buffer import Buffer

from a_configuration.a_base_config.c_models.config_convolutional_models import Config1DConvolutionalModel
from a_configuration.a_base_config.c_models.config_linear_models import ConfigLinearModel
from a_configuration.a_base_config.c_models.config_recurrent_convolutional_models import ConfigRecurrent1DConvolutionalModel
from a_configuration.a_base_config.c_models.config_recurrent_linear_models import ConfigRecurrentLinearModel
from g_utils.types import Transition


class HerEpisodeBuffer:
    def __init__(self, observation_space, action_space, config):
        self.observation_space = observation_space
        self.action_space = action_space
        self.config = config

        self.observations_buffer = None
        self.actions_buffer = None
        self.next_observations_buffer = None
        self.rewards_buffer = None
        self.dones_buffer = None
        self.infos_buffer = None

        self.size = None

    def reset(self):
        self.observations_buffer, \
        self.actions_buffer, \
        self.next_observations_buffer, \
        self.rewards_buffer, \
        self.dones_buffer, \
        self.infos_buffer = Buffer.get_new_buffer_without_capacity(
            observation_space=self.observation_space, action_space=self.action_space, config=self.config
        )
        self.size = 0

    def append(self, transition):
        self.observations_buffer = torch.cat(
            (self.observations_buffer, torch.unsqueeze(transition.observation, dim=0)), dim=0
        )

        if isinstance(self.action_space, Discrete):
            self.actions_buffer = torch.cat(
                (self.actions_buffer, torch.full((1,), fill_value=transition.action)), dim=0
            )
        else:
            self.actions_buffer = torch.cat(
                (self.actions_buffer, torch.unsqueeze(transition.action, dim=0)), dim=0
            )

        self.next_observations_buffer = torch.cat(
            (self.next_observations_buffer, torch.unsqueeze(transition.next_observation, dim=0)), dim=0
        )

        self.rewards_buffer = torch.cat(
            (self.rewards_buffer, torch.full((1,), fill_value=transition.reward)), dim=0
        )

        self.dones_buffer = torch.cat(
            (self.dones_buffer, torch.full((1,), fill_value=transition.done)), dim=0
        )

        self.infos_buffer.append(transition.info)

        self.size += 1

    def size(self):
        return self.size

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

        transitions_in_buffer = zip(
            self.observations_buffer,
            self.actions_buffer,
            self.next_observations_buffer,
            self.rewards_buffer,
            self.dones_buffer,
            self.infos_buffer
        )
        for idx, (observation, action, next_observation, reward, done, info) in enumerate(transitions_in_buffer):
            new_episode_trajectory.append(Transition(
                observation=self._get_observation_and_goal(observation, her_goal),
                action=action,
                next_observation=self._get_observation_and_goal(next_observation, her_goal),
                reward=1.0 if idx == self.size - 1 else 0.0,
                done=True if idx == self.size - 1 else False,
                info=info
            ))

        #print(new_episode_buffer, "!!!")
        return new_episode_trajectory


if __name__ == "__main__":
    class Config:
        BUFFER_CAPACITY = 8
        MODEL_PARAMETER = None
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        USE_PER = False

    config = Config()