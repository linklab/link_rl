import warnings

from gym.spaces import Discrete, Box

warnings.filterwarnings('ignore')
warnings.simplefilter("ignore")

import time
from collections import deque

import torch
import numpy as np

from e_main.supports.actor import Actor
from g_utils.commons import get_train_env, get_single_env, console_log_comparison, wandb_log_comparison, MeanBuffer
from g_utils.types import AgentType, AgentMode, Transition


class LearnerComparison:
    def __init__(self, run, agents, wandb_obj=None, parameter_c=None, comparison_stat=None):
        self.run = run
        self.agents = agents
        self.wandb_obj = wandb_obj
        self.parameter_c = parameter_c

        #######################################
        ### 모든 에이전트가 공통으로 사용하는 파라미터 ###
        #######################################
        self.n_actors = self.parameter_c.N_ACTORS  # SHOULD BE 1
        self.n_vectorized_envs = self.parameter_c.N_VECTORIZED_ENVS
        self.next_train_time_step = self.parameter_c.TRAIN_INTERVAL_GLOBAL_TIME_STEPS
        self.next_test_training_step = self.parameter_c.TEST_INTERVAL_TRAINING_STEPS
        self.next_console_log = self.parameter_c.CONSOLE_LOG_INTERVAL_TRAINING_STEPS
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
        self.is_train_success_done_per_agent = []

        self.comparison_stat = comparison_stat

        for agent_idx, _ in enumerate(agents):
            self.train_envs_per_agent.append(get_train_env(self.parameter_c))
            self.test_envs_per_agent.append(get_single_env(self.parameter_c.AGENT_PARAMETERS[agent_idx]))
            self.episode_rewards_per_agent.append(np.zeros(shape=(self.n_actors, self.n_vectorized_envs)))
            self.episode_reward_buffer_per_agent.append(MeanBuffer(self.parameter_c.N_EPISODES_FOR_MEAN_CALCULATION))

            self.transition_generators_per_agent.append(self.generator_on_policy_transition(agent_idx))

            self.histories_per_agent.append(
                [deque(maxlen=parameter_c.AGENT_PARAMETERS[agent_idx].N_STEP) for _ in range(self.n_vectorized_envs)]
            )

            self.total_episodes_per_agent.append(0)
            self.training_steps_per_agent.append(0)
            self.n_rollout_transitions_per_agent.append(0)
            self.last_mean_episode_reward_per_agent.append(0.0)

            self.is_terminated_per_agent.append(False)
            self.is_train_success_done_per_agent.append(False)

        self.total_time_steps = 0
        self.training_steps = 0
        self.test_idx = 0

        self.train_comparison_start_time = None

    def generator_on_policy_transition(self, agent_idx):
        observations = self.train_envs_per_agent[agent_idx].reset()

        actor_time_step = 0

        while True:
            actor_time_step += 1
            actions = self.agents[agent_idx].get_action(observations)

            if isinstance(self.agents[agent_idx].action_space, Discrete):
                scaled_actions = actions
            elif isinstance(self.agents[agent_idx].action_space, Box):
                scaled_actions = actions * self.agents[agent_idx].action_scale + self.agents[agent_idx].action_bias
            else:
                raise ValueError()

            next_observations, rewards, dones, infos = self.train_envs_per_agent[agent_idx].step(scaled_actions)

            for env_id, (observation, action, next_observation, reward, done, info) in enumerate(
                    zip(observations, actions, next_observations, rewards, dones, infos)
            ):
                info["actor_id"] = 0
                info["env_id"] = env_id
                info["actor_time_step"] = actor_time_step
                self.histories_per_agent[agent_idx][env_id].append(Transition(
                    observation=observation,
                    action=action,
                    next_observation=next_observation,
                    reward=reward,
                    done=done,
                    info=info
                ))

                if len(self.histories_per_agent[agent_idx][env_id]) == self.parameter_c.AGENT_PARAMETERS[agent_idx].N_STEP \
                        or done:
                    n_step_transition = Actor.get_n_step_transition(
                        history=self.histories_per_agent[agent_idx][env_id], env_id=env_id,
                        actor_id=0, info=info, done=done, parameter=self.parameter_c.AGENT_PARAMETERS[agent_idx]
                    )
                    yield n_step_transition

            observations = next_observations

            if self.is_terminated_per_agent[agent_idx]:
                break

        yield None

    def train_comparison_loop(self):
        self.train_comparison_start_time = time.time()

        while True:
            self.total_time_steps += 1

            for agent_idx, _ in enumerate(self.agents):
                n_step_transition = next(self.transition_generators_per_agent[agent_idx])

                self.agents[agent_idx].buffer.append(n_step_transition)
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

                    if self.parameter_c.AGENT_PARAMETERS[agent_idx].AGENT_TYPE == AgentType.REINFORCE:
                        is_train_success_done = self.agents[agent_idx].train(
                            training_steps_v=self.training_steps_per_agent[agent_idx]
                        )
                        if is_train_success_done:
                            self.training_steps_per_agent[agent_idx] += 1
                        self.is_train_success_done_per_agent[agent_idx] = is_train_success_done

            if self.total_time_steps >= self.next_train_time_step:
                for agent_idx, _ in enumerate(self.agents):
                    if self.parameter_c.AGENT_PARAMETERS[agent_idx].AGENT_TYPE != AgentType.REINFORCE:
                        is_train_success_done = self.agents[agent_idx].train(
                            training_steps_v=self.training_steps_per_agent[agent_idx]
                        )
                        if is_train_success_done:
                            self.training_steps_per_agent[agent_idx] += 1
                        self.is_train_success_done_per_agent[agent_idx] = is_train_success_done

                self.next_train_time_step += self.parameter_c.TRAIN_INTERVAL_GLOBAL_TIME_STEPS

                if all(self.is_train_success_done_per_agent):
                    self.training_steps += 1

            if self.training_steps >= self.next_console_log:
                console_log_comparison(
                    self.total_time_steps,
                    self.total_episodes_per_agent,
                    self.last_mean_episode_reward_per_agent,
                    self.n_rollout_transitions_per_agent,
                    self.training_steps_per_agent,
                    self.agents,
                    self.parameter_c
                )
                self.next_console_log += self.parameter_c.CONSOLE_LOG_INTERVAL_TRAINING_STEPS

            if self.training_steps >= self.next_test_training_step:
                for agent_idx, _ in enumerate(self.agents):
                    self.testing(self.run, agent_idx, self.training_steps)

                for agent_idx, _ in enumerate(self.agents):
                    self.update_stat(self.run, agent_idx, self.test_idx)

                if self.parameter_c.USE_WANDB:
                    wandb_log_comparison(
                        run=self.run,
                        agents=self.agents,
                        agent_labels=self.parameter_c.AGENT_LABELS,
                        n_episodes_for_mean_calculation=self.parameter_c.N_EPISODES_FOR_MEAN_CALCULATION,
                        comparison_stat=self.comparison_stat,
                        wandb_obj=self.wandb_obj
                    )

                self.next_test_training_step += self.parameter_c.TEST_INTERVAL_TRAINING_STEPS
                self.test_idx += 1

            if self.training_steps >= self.parameter_c.MAX_TRAINING_STEPS:
                for agent_idx, _ in enumerate(self.agents):
                    print("[TRAIN TERMINATION: AGENT {0}] MAX_TRAINING_STEPS ({1}) REACHES!!!".format(
                        agent_idx,  self.parameter_c.MAX_TRAINING_STEPS
                    ))
                    self.is_terminated_per_agent[agent_idx] = True
                break

    def testing(self, run, agent_idx, training_steps):
        print("*" * 160)

        avg, std = self.play_for_testing(self.parameter_c.N_TEST_EPISODES, agent_idx)

        self.comparison_stat.test_episode_reward_avg_per_agent[run, agent_idx, self.test_idx] = avg
        self.comparison_stat.test_episode_reward_std_per_agent[run, agent_idx, self.test_idx] = std
        self.comparison_stat.mean_episode_reward_per_agent[run, agent_idx, self.test_idx] = \
            self.last_mean_episode_reward_per_agent[agent_idx]

        elapsed_time = time.time() - self.train_comparison_start_time
        formatted_elapsed_time = time.strftime('%H:%M:%S', time.gmtime(elapsed_time))
        print("[Test: {0}, Agent: {1}, Training Step: {2:6,}] "
              "Episode Reward - Average: {3:.3f}, Standard Dev.: {4:.3f}, Elapsed Time: {5} ".format(
            self.test_idx + 1, agent_idx, training_steps, avg, std,
            formatted_elapsed_time
        ))
        print("*" * 160)

    def play_for_testing(self, n_test_episodes, agent_idx):
        episode_reward_lst = []
        for i in range(n_test_episodes):
            episode_reward = 0  # cumulative_reward

            # Environment 초기화와 변수 초기화
            observation = self.test_envs_per_agent[agent_idx].reset()
            observation = np.expand_dims(observation, axis=0)

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

                next_observation, reward, done, _ = self.test_envs_per_agent[agent_idx].step(scaled_action)
                next_observation = np.expand_dims(next_observation, axis=0)

                episode_reward += reward  # episode_reward 를 산출하는 방법은 감가률 고려하지 않는 이 라인이 더 올바름.
                observation = next_observation

                if done:
                    break

            episode_reward_lst.append(episode_reward)

        return np.average(episode_reward_lst), np.std(episode_reward_lst)

    def update_stat(self, run, agent_idx, test_idx):
        # 1
        min_value = np.finfo(np.float64).max
        max_value = np.finfo(np.float64).min
        sum_value = 0.0

        for i in range(run + 1):
            if self.comparison_stat.test_episode_reward_avg_per_agent[i, agent_idx, test_idx] < min_value:
                min_value = self.comparison_stat.test_episode_reward_avg_per_agent[i, agent_idx, test_idx]

            if self.comparison_stat.test_episode_reward_avg_per_agent[i, agent_idx, test_idx] > max_value:
                max_value = self.comparison_stat.test_episode_reward_avg_per_agent[i, agent_idx, test_idx]

            sum_value += self.comparison_stat.test_episode_reward_avg_per_agent[i, agent_idx, test_idx]

        self.comparison_stat.MIN_test_episode_reward_avg_per_agent[agent_idx, test_idx] = min_value
        self.comparison_stat.MAX_test_episode_reward_avg_per_agent[agent_idx, test_idx] = max_value
        self.comparison_stat.MEAN_test_episode_reward_avg_per_agent[agent_idx, test_idx] = sum_value / (run + 1)

        # 2
        min_value = np.finfo(np.float64).max
        max_value = np.finfo(np.float64).min
        sum_value = 0.0

        for i in range(run + 1):
            if self.comparison_stat.test_episode_reward_std_per_agent[i, agent_idx, test_idx] < min_value:
                min_value = self.comparison_stat.test_episode_reward_std_per_agent[i, agent_idx, test_idx]

            if self.comparison_stat.test_episode_reward_std_per_agent[i, agent_idx, test_idx] > max_value:
                max_value = self.comparison_stat.test_episode_reward_std_per_agent[i, agent_idx, test_idx]

            sum_value += self.comparison_stat.test_episode_reward_std_per_agent[i, agent_idx, test_idx]

        self.comparison_stat.MIN_test_episode_reward_std_per_agent[agent_idx, test_idx] = min_value
        self.comparison_stat.MAX_test_episode_reward_std_per_agent[agent_idx, test_idx] = max_value
        self.comparison_stat.MEAN_test_episode_reward_std_per_agent[agent_idx, test_idx] = sum_value / (run + 1)

        # 3
        min_value = np.finfo(np.float64).max
        max_value = np.finfo(np.float64).min
        sum_value = 0.0

        for i in range(run + 1):
            if self.comparison_stat.mean_episode_reward_per_agent[i, agent_idx, test_idx] < min_value:
                min_value = self.comparison_stat.mean_episode_reward_per_agent[i, agent_idx, test_idx]

            if self.comparison_stat.mean_episode_reward_per_agent[i, agent_idx, test_idx] > max_value:
                max_value = self.comparison_stat.mean_episode_reward_per_agent[i, agent_idx, test_idx]

            sum_value += self.comparison_stat.mean_episode_reward_per_agent[i, agent_idx, test_idx]

        self.comparison_stat.MIN_mean_episode_reward_per_agent[agent_idx, test_idx] = min_value
        self.comparison_stat.MAX_mean_episode_reward_per_agent[agent_idx, test_idx] = max_value
        self.comparison_stat.MEAN_mean_episode_reward_per_agent[agent_idx, test_idx] = sum_value / (run + 1)

