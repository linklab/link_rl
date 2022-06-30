import numpy as np
from gym.spaces import Discrete, Box
from gym.vector import VectorEnv

from link_rl.g_utils.commons import get_single_env
from link_rl.g_utils.types import AgentType, AgentMode


class Tester:
    def __init__(self, agent, config):
        self.agent = agent
        self.config = config
        self.test_env = get_single_env(self.config, train_mode=False)

    def play_for_testing(self, n_test_episodes):
        self.agent.model.eval()

        episode_reward_lst = []

        if self.config.CUSTOM_ENV_STAT is not None:
            self.config.CUSTOM_ENV_STAT.test_reset()

        for i in range(n_test_episodes):
            episode_reward = 0  # cumulative_reward
            episode_step = 0

            observation, info = self.test_env.reset(return_info=True)

            done = False

            while not done:
                if not self.config.AGENT_TYPE == AgentType.TDMPC:
                    if not isinstance(self.test_env, VectorEnv):
                        observation = np.expand_dims(observation, axis=0)

                if self.agent.is_recurrent_model:
                    self.agent.model.init_recurrent_hidden()
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

                    if not isinstance(self.test_env, VectorEnv):
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
                    else:
                        scaled_action = action

                next_observation, reward, done, info = self.test_env.step(scaled_action)
                episode_step += 1

                episode_reward += reward
                observation = next_observation

            episode_reward_lst.append(episode_reward)

            if self.config.CUSTOM_ENV_STAT is not None:
                self.config.CUSTOM_ENV_STAT.test_episode_done(info=info)

        if self.config.CUSTOM_ENV_STAT is not None:
            self.config.CUSTOM_ENV_STAT.test_evaluate()

        self.agent.model.train()

        return min(episode_reward_lst)
