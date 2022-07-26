import time

import numpy as np
from gym.spaces import Discrete, Box

from link_rl.a_configuration.a_base_config.a_environments.competition_olympics import ConfigCompetitionOlympics
from link_rl.a_configuration.a_base_config.a_environments.open_ai_gym.config_gym_atari import ConfigGymAtari
from link_rl.a_configuration.a_base_config.a_environments.open_ai_gym.config_gym_mujoco import ConfigMujoco
from link_rl.a_configuration.a_base_config.a_environments.somo_gym import ConfigSomoGym
from link_rl.h_utils.commons import get_single_env
from link_rl.h_utils.types import AgentType, AgentMode


class Tester:
    def __init__(self, agent, config, play=False):
        self.agent = agent
        self.config = config
        self.play = play

        if isinstance(config, ConfigSomoGym):
            self.max_episode_step = 1_000
        else:
            self.max_episode_step = None

        if isinstance(self.config, ConfigCompetitionOlympics):
            self.test_env = get_single_env(self.config, play=self.play, agent=self.agent)
        else:
            self.test_env = get_single_env(self.config, play=self.play)

    # def episode_continue(self, done, episode_step):
    #     if isinstance(self.config, ConfigSomoGym):
    #         return episode_step < self.max_episode_step
    #     else:
    #         return not done

    def episode_continue(self, done, episode_step):
        return not done

    def play_for_testing(self, n_episodes, delay=0.0):
        self.agent.model.eval()

        episode_reward_lst = []
        episode_step_lst = []

        if self.config.CUSTOM_ENV_STAT is not None:
            self.config.CUSTOM_ENV_STAT.test_reset()

        for i in range(n_episodes):
            episode_reward = 0  # cumulative_reward
            episode_step = 0

            render_before_reset_conditions = [
                self.play,
                not isinstance(self.config, ConfigGymAtari),
                isinstance(self.config, ConfigMujoco)
            ]

            render_after_reset_conditions = [
                self.play,
                not isinstance(self.config, ConfigGymAtari),
                not isinstance(self.config, ConfigMujoco)
            ]

            if all(render_before_reset_conditions):
                self.test_env.render()

            observation, info = self.test_env.reset(return_info=True)

            if self.agent.is_recurrent_model:
                self.agent.model.init_recurrent_hidden()

            if all(render_after_reset_conditions):
                self.test_env.render()

            done = False

            while self.episode_continue(done, episode_step):
                if not self.config.AGENT_TYPE == AgentType.TDMPC:
                    observation = np.expand_dims(observation, axis=0)

                if self.agent.is_recurrent_model:
                    observation = [(observation, self.agent.model.recurrent_hidden)]

                if self.config.ACTION_MASKING:
                    unavailable_actions = [info['unavailable_actions']]
                else:
                    unavailable_actions = None

                if self.config.AGENT_TYPE == AgentType.TDMPC:
                    action = self.agent.get_action(
                        obs=observation, mode=AgentMode.TEST, step=episode_step, t0=episode_step == 0
                    )
                    scaled_action = action
                    # scaled_action = scaled_action.cpu().numpy()
                else:
                    if self.config.ACTION_MASKING:
                        action = self.agent.get_action(
                            obs=observation, unavailable_actions=unavailable_actions, mode=AgentMode.TEST
                        )
                    else:
                        action = self.agent.get_action(
                            obs=observation, mode=AgentMode.TEST
                        )

                    if isinstance(self.agent.action_space, Discrete):
                        if action.ndim == 0:
                            scaled_action = action
                        elif action.ndim == 1:
                            scaled_action = action[0]
                        else:
                            raise ValueError()
                    elif isinstance(self.agent.action_space, Box):
                        if action.ndim == 1:
                            if self.agent.action_scale is not None:
                                scaled_action = action * self.agent.action_scale + self.agent.action_bias
                            else:
                                scaled_action = action
                        elif action.ndim == 2:
                            if self.agent.action_scale is not None:
                                scaled_action = action[0] * self.agent.action_scale + self.agent.action_bias
                            else:
                                scaled_action = action[0]
                        else:
                            raise ValueError()
                    else:
                        raise ValueError()

                next_observation, reward, done, info = self.test_env.step(scaled_action)

                episode_step += 1

                episode_reward += reward
                observation = next_observation

                if self.play:
                    if not isinstance(self.config, ConfigGymAtari):
                        self.test_env.render()
                    time.sleep(delay)

            episode_reward_lst.append(episode_reward)
            episode_step_lst.append(episode_step)

            if self.play:
                print("[EPISODE: {0}] EPISODE_STEPS: {1:3d}, EPISODE REWARD: {2:4.1f}".format(
                    i + 1, episode_step, episode_reward
                ))
            else:
                if self.config.CUSTOM_ENV_STAT is not None:
                    self.config.CUSTOM_ENV_STAT.test_episode_done(info=info)

        if not self.play and self.config.CUSTOM_ENV_STAT is not None:
            self.config.CUSTOM_ENV_STAT.test_evaluate()

        self.agent.model.train()

        test_episode_reward_mean = sum(episode_reward_lst) / len(episode_reward_lst)

        min_idx_lst = [i for i, val in enumerate(episode_reward_lst) if val == test_episode_reward_mean]

        episode_reward_min_step_sum = 0.0
        for i in min_idx_lst:
            episode_reward_min_step_sum += episode_step_lst[i]

        test_episode_reward_min_step = episode_reward_min_step_sum / len(min_idx_lst)

        return test_episode_reward_mean, test_episode_reward_min_step
