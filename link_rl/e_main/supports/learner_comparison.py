import warnings

from link_rl.a_configuration.a_base_config.c_models.config_recurrent_convolutional_models import \
    ConfigRecurrent2DConvolutionalModel, ConfigRecurrent1DConvolutionalModel

warnings.filterwarnings('ignore')
warnings.simplefilter("ignore")

import time
from collections import deque
import numpy as np
from gym.spaces import Discrete, Box

from link_rl.a_configuration.a_base_config.c_models.config_recurrent_linear_models import ConfigRecurrentLinearModel
from link_rl.e_main.supports.actor import Actor
from link_rl.g_utils.commons import get_train_env, get_single_env, console_log_comparison, wandb_log_comparison, MeanBuffer
from link_rl.g_utils.types import AgentMode, Transition, OnPolicyAgentTypes, OffPolicyAgentTypes, AgentType


class LearnerComparison:
    def __init__(self, run, agents, wandb_obj=None, config_c=None, comparison_stat=None):
        self.run = run
        self.agents = agents
        self.wandb_obj = wandb_obj
        self.config_c = config_c

        #######################################
        ### 모든 에이전트가 공통으로 사용하는 파라미터 ###
        #######################################
        self.n_actors = self.config_c.N_ACTORS  # SHOULD BE 1
        self.n_vectorized_envs = self.config_c.N_VECTORIZED_ENVS
        self.next_train_time_step = self.config_c.TRAIN_INTERVAL_GLOBAL_TIME_STEPS
        #######################################

        self.train_envs_per_agent = []
        self.test_envs_per_agent = []
        self.episode_rewards_per_agent = []
        self.episode_reward_buffer_per_agent = []
        self.transition_generators_per_agent = []
        self.histories_per_agent = []

        self.total_episodes_per_agent = []
        self.training_steps_per_agent = []
        self.n_rollout_transitions_per_agent = []
        self.last_mean_episode_reward_per_agent = []
        self.last_loss_train_per_agent = []
        self.is_terminated_per_agent = []
        self.is_recurrent_model_per_agent = []

        self.next_console_log_per_agent = []
        self.next_test_training_step_per_agent = []
        self.test_idx_per_agent = []

        self.comparison_stat = comparison_stat

        for agent_idx, _ in enumerate(agents):
            if self.config_c.AGENT_PARAMETERS[agent_idx].AGENT_TYPE == AgentType.TDMPC:
                self.train_envs_per_agent.append(
                    get_single_env(self.config_c.AGENT_PARAMETERS[agent_idx], train_mode=True)
                )
            else:
                self.train_envs_per_agent.append(get_train_env(self.config_c.AGENT_PARAMETERS[agent_idx]))

            self.test_envs_per_agent.append(get_single_env(self.config_c.AGENT_PARAMETERS[agent_idx], train_mode=False))

            self.episode_rewards_per_agent.append(np.zeros(shape=(self.n_actors, self.n_vectorized_envs)))
            self.episode_reward_buffer_per_agent.append(MeanBuffer(self.config_c.N_EPISODES_FOR_MEAN_CALCULATION))

            self.transition_generators_per_agent.append(self.generate_transition_for_comparison(agent_idx))

            self.histories_per_agent.append(
                [deque(maxlen=config_c.AGENT_PARAMETERS[agent_idx].N_STEP) for _ in range(self.n_vectorized_envs)]
            )

            self.total_episodes_per_agent.append(0)
            self.training_steps_per_agent.append(0)
            self.n_rollout_transitions_per_agent.append(0)
            self.last_mean_episode_reward_per_agent.append(0.0)

            if self.config_c.CUSTOM_ENV_COMPARISON_STAT is not None:
                self.config_c.CUSTOM_ENV_COMPARISON_STAT.train_reset()

            self.is_terminated_per_agent.append(False)

            self.is_recurrent_model_per_agent.append(any([
                isinstance(
                    self.config_c.AGENT_PARAMETERS[agent_idx].MODEL_PARAMETER, ConfigRecurrentLinearModel
                ),
                isinstance(
                    self.config_c.AGENT_PARAMETERS[agent_idx].MODEL_PARAMETER, ConfigRecurrent1DConvolutionalModel
                ),
                isinstance(
                    self.config_c.AGENT_PARAMETERS[agent_idx].MODEL_PARAMETER, ConfigRecurrent2DConvolutionalModel
                )
            ]))

            self.next_console_log_per_agent.append(self.config_c.CONSOLE_LOG_INTERVAL_TRAINING_STEPS)
            self.next_test_training_step_per_agent.append(self.config_c.TEST_INTERVAL_TRAINING_STEPS)
            self.test_idx_per_agent.append(0)

        self.total_time_step = 0

        self.train_comparison_start_time = None

    def generate_transition_for_comparison(self, agent_idx):
        observations, infos = self.train_envs_per_agent[agent_idx].reset(return_info=True)

        if self.is_recurrent_model_per_agent[agent_idx]:
            self.agents[agent_idx].model.init_recurrent_hidden()
            observations = [(observations, self.agents[agent_idx].model.recurrent_hidden)]

        while True:
            actions = self.agents[agent_idx].get_action(observations)

            if isinstance(self.agents[agent_idx].action_space, Discrete):
                scaled_actions = actions
            elif isinstance(self.agents[agent_idx].action_space, Box):
                scaled_actions = actions * self.agents[agent_idx].action_scale + self.agents[agent_idx].action_bias
            else:
                raise ValueError()

            next_observations, rewards, dones, infos = self.train_envs_per_agent[agent_idx].step(scaled_actions)

            if self.is_recurrent_model_per_agent[agent_idx]:
                next_observations = [(next_observations, self.agents[agent_idx].model.recurrent_hidden)]

            for env_id, (observation, action, next_observation, reward, done, info) in enumerate(
                    zip(observations, actions, next_observations, rewards, dones, infos)
            ):
                info["actor_id"] = 0
                info["env_id"] = env_id
                self.histories_per_agent[agent_idx][env_id].append(Transition(
                    observation=observation,
                    action=action,
                    next_observation=next_observation,
                    reward=reward,
                    done=done,
                    info=info
                ))

                if len(self.histories_per_agent[agent_idx][env_id]) == self.config_c.AGENT_PARAMETERS[agent_idx].N_STEP \
                        or done:
                    n_step_transition = Actor.get_n_step_transition(
                        history=self.histories_per_agent[agent_idx][env_id], env_id=env_id,
                        actor_id=0, info=info, done=done, config=self.config_c.AGENT_PARAMETERS[agent_idx]
                    )
                    yield n_step_transition

            observations = next_observations

            if self.is_terminated_per_agent[agent_idx]:
                break

        yield None

    def train_comparison_loop(self, run):
        self.train_comparison_start_time = time.time()

        while True:
            if all([is_terminated for is_terminated in self.is_terminated_per_agent]):
                break

            self.total_time_step += 1

            for agent_idx, _ in enumerate(self.agents):
                if not self.is_terminated_per_agent[agent_idx]:
                    n_step_transition = next(self.transition_generators_per_agent[agent_idx])

                    if self.config_c.AGENT_PARAMETERS[agent_idx].AGENT_TYPE in OnPolicyAgentTypes:
                        self.agents[agent_idx].buffer.append(n_step_transition)
                    elif self.config_c.AGENT_PARAMETERS[agent_idx].AGENT_TYPE in OffPolicyAgentTypes:
                        self.agents[agent_idx].replay_buffer.append(n_step_transition)
                    else:
                        raise ValueError()

                    self.n_rollout_transitions_per_agent[agent_idx] += 1

                    actor_id = n_step_transition.info["actor_id"]   # SHOULD BE 1
                    env_id = n_step_transition.info["env_id"]
                    self.episode_rewards_per_agent[agent_idx][actor_id][env_id] += n_step_transition.reward

                    if n_step_transition.done:
                        self.total_episodes_per_agent[agent_idx] += 1

                        self.episode_reward_buffer_per_agent[agent_idx].add(
                            self.episode_rewards_per_agent[agent_idx][actor_id][env_id]
                        )
                        self.last_mean_episode_reward_per_agent[agent_idx] = \
                            self.episode_reward_buffer_per_agent[agent_idx].mean()

                        self.episode_rewards_per_agent[agent_idx][actor_id][env_id] = 0.0

                        if self.config_c.CUSTOM_ENV_COMPARISON_STAT is not None:
                            self.config_c.CUSTOM_ENV_COMPARISON_STAT.train_evaluate(
                                agent_idx=agent_idx, last_train_env_info=n_step_transition.info
                            )

            if self.total_time_step >= self.next_train_time_step:
                for agent_idx, _ in enumerate(self.agents):
                    if not self.is_terminated_per_agent[agent_idx]:
                        count_training_steps = self.agents[agent_idx].train(
                            training_steps_v=self.training_steps_per_agent[agent_idx]
                        )
                        self.training_steps_per_agent[agent_idx] += count_training_steps

                        if self.training_steps_per_agent[agent_idx] >= self.next_console_log_per_agent[agent_idx]:
                            console_log_comparison(
                                run=self.run,
                                total_time_step=self.total_time_step,
                                total_episodes_per_agent=self.total_episodes_per_agent,
                                last_mean_episode_reward_per_agent=self.last_mean_episode_reward_per_agent,
                                n_rollout_transitions_per_agent=self.n_rollout_transitions_per_agent,
                                training_steps_per_agent=self.training_steps_per_agent,
                                agents=self.agents,
                                config_c=self.config_c
                            )
                            self.next_console_log_per_agent[agent_idx] += self.config_c.CONSOLE_LOG_INTERVAL_TRAINING_STEPS

                        if self.training_steps_per_agent[agent_idx] >= self.next_test_training_step_per_agent[agent_idx]:
                            self.testing(self.run, agent_idx, self.training_steps_per_agent[agent_idx])
                            self.update_stat(self.run, agent_idx, self.test_idx_per_agent[agent_idx])

                            if self.config_c.USE_WANDB:
                                wandb_log_comparison(
                                    run=self.run,
                                    training_steps_per_agent=self.training_steps_per_agent,
                                    agents=self.agents,
                                    agent_labels=self.config_c.AGENT_LABELS,
                                    n_episodes_for_mean_calculation=self.config_c.N_EPISODES_FOR_MEAN_CALCULATION,
                                    comparison_stat=self.comparison_stat,
                                    wandb_obj=self.wandb_obj,
                                    config_c=self.config_c
                                )

                            self.next_test_training_step_per_agent[agent_idx] += self.config_c.TEST_INTERVAL_TRAINING_STEPS
                            self.test_idx_per_agent[agent_idx] += 1

                        if self.training_steps_per_agent[agent_idx] >= self.config_c.MAX_TRAINING_STEPS:
                            print("[TRAIN TERMINATION: AGENT {0}] MAX_TRAINING_STEPS ({1}) REACHES!!!".format(
                                agent_idx,  self.config_c.MAX_TRAINING_STEPS
                            ))
                            self.is_terminated_per_agent[agent_idx] = True

                self.next_train_time_step += self.config_c.TRAIN_INTERVAL_GLOBAL_TIME_STEPS

    def testing(self, run, agent_idx, training_step):
        avg, std = self.play_for_testing(self.config_c.N_TEST_EPISODES, run, agent_idx)

        self.comparison_stat.test_episode_reward_avg_per_agent[run, agent_idx, self.test_idx_per_agent[agent_idx]] = avg
        self.comparison_stat.test_episode_reward_std_per_agent[run, agent_idx, self.test_idx_per_agent[agent_idx]] = std
        self.comparison_stat.mean_episode_reward_per_agent[run, agent_idx, self.test_idx_per_agent[agent_idx]] = \
            self.last_mean_episode_reward_per_agent[agent_idx]

        elapsed_time = time.time() - self.train_comparison_start_time
        formatted_elapsed_time = time.strftime('%H:%M:%S', time.gmtime(elapsed_time))

        print("*" * 160)
        test_str = "[Test: {0}, Agent: {1}, Training Step: {2:6,}] " \
                   "Episode Reward - Average: {3:.3f}, Standard Dev.: {4:.3f}, Elapsed Time: {5} ".format(
            self.test_idx_per_agent[agent_idx] + 1, agent_idx, training_step, avg, std, formatted_elapsed_time
        )

        if self.config_c.CUSTOM_ENV_COMPARISON_STAT is not None:
            test_str += "," + self.config_c.CUSTOM_ENV_COMPARISON_STAT.test_evaluation_str()

        print(test_str)

        print("*" * 160)

    def play_for_testing(self, n_test_episodes, run, agent_idx):
        self.agents[agent_idx].model.eval()

        episode_reward_lst = []

        if self.config_c.CUSTOM_ENV_COMPARISON_STAT is not None:
            self.config_c.CUSTOM_ENV_COMPARISON_STAT.test_reset()

        for i in range(n_test_episodes):
            episode_reward = 0  # cumulative_reward

            # Environment 초기화와 변수 초기화
            observation = self.test_envs_per_agent[agent_idx].reset()
            observation = np.expand_dims(observation, axis=0)

            if self.is_recurrent_model_per_agent[agent_idx]:
                self.agents[agent_idx].model.init_recurrent_hidden()
                observation = [(observation, self.agents[agent_idx].model.recurrent_hidden)]

            while True:
                action = self.agents[agent_idx].get_action(observation, mode=AgentMode.TEST)

                if isinstance(self.agents[agent_idx].action_space, Discrete):
                    if action.ndim == 0:
                        scaled_action = action
                    elif action.ndim == 1:
                        scaled_action = action[0]
                    else:
                        raise ValueError()
                elif isinstance(self.agents[agent_idx].action_space, Box):
                    if action.ndim == 1:
                        if self.agents[agent_idx].action_scale is not None:
                            scaled_action = action * self.agents[agent_idx].action_scale[0] + self.agents[agent_idx].action_bias[0]
                        else:
                            scaled_action = action
                    elif action.ndim == 2:
                        if self.agents[agent_idx].action_scale is not None:
                            scaled_action = action[0] * self.agents[agent_idx].action_scale[0] + self.agents[agent_idx].action_bias[0]
                        else:
                            scaled_action = action[0]
                    else:
                        raise ValueError()
                else:
                    raise ValueError()

                next_observation, reward, done, info = self.test_envs_per_agent[agent_idx].step(scaled_action)

                next_observation = np.expand_dims(next_observation, axis=0)

                if self.is_recurrent_model_per_agent[agent_idx]:
                    next_observation = [(next_observation, self.agents[agent_idx].model.recurrent_hidden)]

                episode_reward += reward  # episode_reward 를 산출하는 방법은 감가률 고려하지 않는 이 라인이 더 올바름.
                observation = next_observation

                if done:
                    break

            episode_reward_lst.append(episode_reward)

            if self.config_c.CUSTOM_ENV_COMPARISON_STAT is not None:
                self.config_c.CUSTOM_ENV_COMPARISON_STAT.test_episode_done(info)

        self.agents[agent_idx].model.train()

        if self.config_c.CUSTOM_ENV_COMPARISON_STAT is not None:
            self.config_c.CUSTOM_ENV_COMPARISON_STAT.test_evaluate()

        return np.average(episode_reward_lst), np.std(episode_reward_lst)

    def update_stat(self, run, agent_idx, test_idx):
        target_stats = [
            self.comparison_stat.test_episode_reward_avg_per_agent,
            self.comparison_stat.test_episode_reward_std_per_agent,
            self.comparison_stat.mean_episode_reward_per_agent,
        ]

        min_target_stats = [
            self.comparison_stat.MIN_test_episode_reward_avg_per_agent,
            self.comparison_stat.MIN_test_episode_reward_std_per_agent,
            self.comparison_stat.MIN_mean_episode_reward_per_agent
        ]

        max_target_stats = [
            self.comparison_stat.MAX_test_episode_reward_avg_per_agent,
            self.comparison_stat.MAX_test_episode_reward_std_per_agent,
            self.comparison_stat.MAX_mean_episode_reward_per_agent
        ]

        mean_target_stats = [
            self.comparison_stat.MEAN_test_episode_reward_avg_per_agent,
            self.comparison_stat.MEAN_test_episode_reward_std_per_agent,
            self.comparison_stat.MEAN_mean_episode_reward_per_agent
        ]

        if self.config_c.CUSTOM_ENV_COMPARISON_STAT is not None:
            self.config_c.CUSTOM_ENV_COMPARISON_STAT.update_stat(
                run=run,
                agent_idx=agent_idx,
                stat_idx=self.test_idx_per_agent[agent_idx],
                target_stats=target_stats,
                min_target_stats=min_target_stats,
                max_target_stats=max_target_stats,
                mean_target_stats=mean_target_stats
            )

        for target_stat, min_target_stat, max_target_stat, mean_target_stat in zip(
                target_stats, min_target_stats, max_target_stats, mean_target_stats
        ):
            min_value = np.finfo(np.float64).max
            max_value = np.finfo(np.float64).min
            sum_value = 0.0

            for i in range(run + 1):
                if target_stat[i, agent_idx, test_idx] < min_value:
                    min_value = target_stat[i, agent_idx, test_idx]

                if target_stat[i, agent_idx, test_idx] > max_value:
                    max_value = target_stat[i, agent_idx, test_idx]

                sum_value += target_stat[i, agent_idx, test_idx]

            min_target_stat[agent_idx, test_idx] = min_value
            max_target_stat[agent_idx, test_idx] = max_value
            mean_target_stat[agent_idx, test_idx] = sum_value / (run + 1)
