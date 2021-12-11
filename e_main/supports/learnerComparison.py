from collections import deque

import torch
import torch.multiprocessing as mp
import numpy as np
import time

from e_main.supports.actor import Actor
from g_utils.commons import model_save, console_log, wandb_log, get_wandb_obj, get_train_env, get_single_env, \
    console_log_comparison, wandb_log_comparison
from g_utils.buffers import Buffer
from g_utils.types import AgentType, AgentMode, Transition


class LearnerComparison:
    def __init__(self, n_agents, agents, device=torch.device("cpu"), parameter_comparison=None):
        self.n_agents = n_agents
        self.agents = agents
        self.device = device
        self.parameter_comparison = parameter_comparison

        #######################################
        ### 모든 에이전트가 공통으로 사용하는 파라미터 ###
        #######################################
        self.n_actors = self.parameter_comparison.N_ACTORS # SHOULD BE 1
        self.n_vectorized_envs = self.parameter_comparison.N_VECTORIZED_ENVS
        self.next_train_time_step = self.parameter_comparison.TRAIN_INTERVAL_TOTAL_TIME_STEPS
        self.next_test_training_step = self.parameter_comparison.TEST_INTERVAL_TRAINING_STEPS
        self.next_console_log = self.parameter_comparison.CONSOLE_LOG_INTERVAL_TOTAL_TIME_STEPS
        self.n_episodes_for_mean_calculation = self.parameter_comparison.N_EPISODES_FOR_MEAN_CALCULATION
        #######################################

        self.train_envs_per_agent = []
        self.test_envs_per_agent = []
        self.episode_rewards_per_agent = []
        self.episode_reward_lst_per_agent = []
        self.buffers_per_agent = []
        self.transition_generators_per_agent = []
        self.histories_per_agent = []
        for agent_idx in range(n_agents):

            self.train_envs_per_agent.append(
                get_train_env(self.parameter_comparison.parameters[agent_idx])
            )
            self.test_envs_per_agent.append(get_single_env(self.parameter_comparison.parameters[agent_idx]))

            self.episode_rewards_per_agent.append(np.zeros(shape=(self.n_actors, self.n_vectorized_envs)))
            self.episode_reward_lst_per_agent.append([])
            self.buffers_per_agent.append(
                Buffer(
                    capacity=parameter_comparison.parameters[agent_idx].BUFFER_CAPACITY, device=self.device
                )
            )

            self.histories_per_agent.append([])
            for _ in range(self.n_vectorized_envs):
                self.histories_per_agent[agent_idx].append(
                    deque(maxlen=self.parameter_comparison.parameters[agent_idx].N_STEP)
                )

            self.transition_generators_per_agent.append(
                self.generator_on_policy_transition(agent_idx)
            )

        self.n_actor_terminations_per_agent = [0] * self.n_agents

        self.total_time_steps_per_agent = [0] * self.n_agents
        self.total_episodes_per_agent = [0] * self.n_agents
        self.training_steps_per_agent = [0] * self.n_agents
        self.n_rollout_transitions_per_agent = [0] * self.n_agents

        self.last_mean_episode_reward_per_agent = [0.0] * self.n_agents

        self.is_terminated_per_agent = [False] * self.n_agents

        self.test_episode_reward_avg_per_agent = [0.0] * self.n_agents
        self.test_episode_reward_std_per_agent = [0.0] * self.n_agents

    def generator_on_policy_transition(self, agent_idx):
        observations = self.train_envs_per_agent[agent_idx].reset()

        actor_time_step = 0

        while True:
            actor_time_step += 1
            actions = self.agents[agent_idx].get_action(observations)
            next_observations, rewards, dones, infos = self.train_env.step(actions)

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

                if len(self.histories_per_agent[agent_idx][env_id]) == self.parameter_comparison.parameters[agent_idx].N_STEP \
                        or done:
                    n_step_transition = Actor.get_n_step_transition(
                        history=self.histories_per_agent[agent_idx][env_id], env_id=env_id,
                        actor_id=0, info=info, done=done, parameter=self.parameter_comparison.parameters[agent_idx]
                    )
                    yield n_step_transition

            observations = next_observations

            if self.is_terminated_per_agent[agent_idx]:
                break

        yield None

    def train_loop(self):
        if self.parameter_comparison.USE_WANDB:
            wandb_obj = get_wandb_obj(self.parameter_comparison)
        else:
            wandb_obj = None

        global_time_steps = 0
        while True:
            global_time_steps += 1
            for agent_idx in range(self.n_agents):
                n_step_transition = next(self.transition_generators_per_agent[agent_idx])

                if n_step_transition is None:
                    self.n_actor_terminations_per_agent[agent_idx] += 1
                    if self.n_actor_terminations_per_agent[agent_idx] >= self.n_actors:
                        self.is_terminated_per_agent[agent_idx] = True
                        break
                    else:
                        continue
                else:
                    if self.is_terminated_per_agent[agent_idx]:
                        continue
                    else:
                        self.total_time_steps_per_agent[agent_idx] += 1

                self.buffers_per_agent[agent_idx].append(n_step_transition)
                self.n_rollout_transitions_per_agent[agent_idx] += 1

                actor_id = n_step_transition.info["actor_id"] # SHOULD BE 1
                env_id = n_step_transition.info["env_id"]
                self.episode_rewards_per_agent[agent_idx][actor_id][env_id] += n_step_transition.reward

                if self.total_time_steps_per_agent[agent_idx] >= self.next_train_time_step:
                    if self.parameter_comparison.parameters[agent_idx].AGENT_TYPE != AgentType.Reinforce:
                        self.agents[agent_idx].train(
                            buffer=self.buffers_per_agent[agent_idx],
                            training_steps=self.training_steps_per_agent[agent_idx]
                        )
                    self.next_train_time_step += self.parameter_comparison.TRAIN_INTERVAL_TOTAL_TIME_STEPS

                if n_step_transition.done:
                    self.total_episodes_per_agent[agent_idx] += 1

                    self.episode_reward_lst_per_agent[agent_idx].append(
                        self.episode_rewards_per_agent[agent_idx][actor_id][env_id]
                    )
                    self.last_mean_episode_reward_per_agent[agent_idx] = float(np.mean(
                        self.episode_reward_lst_per_agent[agent_idx][-1 * self.n_episodes_for_mean_calculation:]
                    ))

                    self.episode_rewards_per_agent[agent_idx][actor_id][env_id] = 0.0

                    if self.parameter_comparison.parameters[agent_idx].AGENT_TYPE == AgentType.Reinforce:
                        self.agents[agent_idx].train(
                            buffer=self.buffers_per_agent[agent_idx],
                            training_steps=self.training_steps_per_agent[agent_idx]
                        )

                if self.training_steps_per_agent[agent_idx] >= self.next_test_training_step:
                    self.testing(agent_idx)
                    self.next_test_training_step += self.parameter_comparison.TEST_INTERVAL_TRAINING_STEPS
                    if self.parameter_comparison.USE_WANDB:
                        wandb_log_comparison(self, wandb_obj)

                if self.training_steps_per_agent[agent_idx] >= self.parameter_comparison.MAX_TRAINING_STEPS:
                    print("[TRAIN TERMINATION: AGENT {0}] MAX_TRAINING_STEPS ({1}) REACHES!!!".format(
                        agent_idx,  self.parameter_comparison.MAX_TRAINING_STEPS
                    ))
                    self.is_terminated_per_agent[agent_idx] = True

            if global_time_steps >= self.next_console_log:
                console_log_comparison(
                    self.total_episodes_per_agent,
                    self.total_time_steps_per_agent,
                    self.last_mean_episode_reward_per_agent,
                    self.n_rollout_transitions_per_agent,
                    self.training_steps_per_agent,
                    self.agents,
                    self.parameter_comparison
                )
                self.next_console_log += self.parameter_comparison.CONSOLE_LOG_INTERVAL_TOTAL_TIME_STEPS

        if self.parameter_comparison.USE_WANDB:
            wandb_obj.join()

    def testing(self, agent_idx):
        print("*" * 80)
        self.test_episode_reward_avg_per_agent[agent_idx], \
        self.test_episode_reward_std_per_agent[agent_idx] = \
            self.play_for_testing(self.parameter_comparison.N_TEST_EPISODES, agent_idx)

        print("[Test Episode Reward: Agent {0}] Average: {1:.3f}, Standard Dev.: {2:.3f}".format(
            agent_idx,
            self.test_episode_reward_avg_per_agent[agent_idx],
            self.test_episode_reward_std_per_agent[agent_idx]
        ))
        print("*" * 80)

    def play_for_testing(self, n_test_episodes, agent_idx):
        episode_reward_lst = []
        for i in range(n_test_episodes):
            episode_reward = 0  # cumulative_reward

            # Environment 초기화와 변수 초기화
            observation = self.test_envs_per_agent[agent_idx].reset()
            observation = np.expand_dims(observation, axis=0)

            while True:
                action = self.agents[agent_idx].get_action(observation, mode=AgentMode.TEST)

                # action을 통해서 next_state, reward, done, info를 받아온다
                next_observation, reward, done, _ = self.test_envs_per_agent[agent_idx].step(action[0])
                next_observation = np.expand_dims(next_observation, axis=0)

                episode_reward += reward  # episode_reward 를 산출하는 방법은 감가률 고려하지 않는 이 라인이 더 올바름.
                observation = next_observation

                if done:
                    break

            episode_reward_lst.append(episode_reward)

        return np.average(episode_reward_lst), np.std(episode_reward_lst)
