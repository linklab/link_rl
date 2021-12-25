import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter("ignore")

from collections import deque

import torch
import torch.multiprocessing as mp
import numpy as np
import time

from e_main.supports.actor import Actor
from g_utils.commons import model_save, console_log, wandb_log, get_wandb_obj, get_train_env, get_single_env, MeanBuffer
from g_utils.buffers import Buffer
from g_utils.types import AgentType, AgentMode, Transition


class Learner(mp.Process):
    def __init__(self, agent, queue, device=torch.device("cpu"), parameter=None):
        super(Learner, self).__init__()
        self.agent = agent
        self.queue = queue
        self.device = device
        self.parameter = parameter

        self.train_env = None
        self.test_env = None

        self.n_actors = self.parameter.N_ACTORS
        self.n_vectorized_envs = self.parameter.N_VECTORIZED_ENVS
        self.n_actor_terminations = 0

        self.episode_rewards = np.zeros(shape=(self.n_actors, self.n_vectorized_envs))
        self.episode_reward_buffer = MeanBuffer(self.parameter.N_EPISODES_FOR_MEAN_CALCULATION)

        self.total_time_steps = mp.Value('i', 0)
        self.total_episodes = mp.Value('i', 0)
        self.training_steps = mp.Value('i', 0)

        self.n_rollout_transitions = mp.Value('i', 0)

        self.train_start_time = None
        self.last_mean_episode_reward = mp.Value('d', 0.0)

        self.is_terminated = mp.Value('i', False)

        self.test_episode_reward_avg = mp.Value('d', 0.0)
        self.test_episode_reward_std = mp.Value('d', 0.0)

        self.next_train_time_step = parameter.TRAIN_INTERVAL_GLOBAL_TIME_STEPS
        self.next_test_training_step = parameter.TEST_INTERVAL_TRAINING_STEPS
        self.next_console_log = parameter.CONSOLE_LOG_INTERVAL_TRAINING_STEPS

        self.test_idx = mp.Value('i', 0)

        if queue is None: # Sequential
            self.transition_generator = self.generator_on_policy_transition()

            self.histories = []
            for _ in range(self.parameter.N_VECTORIZED_ENVS):
                self.histories.append(deque(maxlen=self.parameter.N_STEP))

    def generator_on_policy_transition(self):
        observations = self.train_env.reset()

        actor_time_step = 0

        while True:
            actor_time_step += 1
            actions = self.agent.get_action(observations)
            next_observations, rewards, dones, infos = self.train_env.step(actions)

            for env_id, (observation, action, next_observation, reward, done, info) in enumerate(
                    zip(observations, actions, next_observations, rewards, dones, infos)
            ):
                info["actor_id"] = 0
                info["env_id"] = env_id
                info["actor_time_step"] = actor_time_step
                self.histories[env_id].append(Transition(
                    observation=observation,
                    action=action,
                    next_observation=next_observation,
                    reward=reward,
                    done=done,
                    info=info
                ))

                if len(self.histories[env_id]) == self.parameter.N_STEP or done:
                    n_step_transition = Actor.get_n_step_transition(
                        history=self.histories[env_id], env_id=env_id,
                        actor_id=0, info=info, done=done, parameter=self.parameter
                    )
                    yield n_step_transition

            observations = next_observations

            if self.is_terminated.value:
                break

        yield None

    def train_loop(self, parallel=False):
        if not parallel:  # parallel인 경우 actor에서 train_env 생성/관리
            self.train_env = get_train_env(self.parameter)

        self.test_env = get_single_env(self.parameter)

        if self.parameter.USE_WANDB:
            wandb_obj = get_wandb_obj(self.parameter, self.agent)
        else:
            wandb_obj = None

        self.train_start_time = time.time()

        while True:
            if parallel:
                n_step_transition = self.queue.get()
            else:
                n_step_transition = next(self.transition_generator)

            self.total_time_steps.value += 1

            if n_step_transition is None:
                self.n_actor_terminations += 1
                if self.n_actor_terminations >= self.n_actors:
                    self.is_terminated.value = True
                    break
                else:
                    continue
            else:
                if self.is_terminated.value:
                    continue

            self.agent.buffer.append(n_step_transition)
            self.n_rollout_transitions.value += 1

            actor_id = n_step_transition.info["actor_id"]
            env_id = n_step_transition.info["env_id"]
            self.episode_rewards[actor_id][env_id] += n_step_transition.reward

            if n_step_transition.done:
                self.total_episodes.value += 1

                self.episode_reward_buffer.add(self.episode_rewards[actor_id][env_id])
                self.last_mean_episode_reward.value = self.episode_reward_buffer.mean()

                self.episode_rewards[actor_id][env_id] = 0.0

                if self.parameter.AGENT_TYPE == AgentType.Reinforce:
                    is_train_success_done = self.agent.train(
                        training_steps_v=self.training_steps.value
                    )
                    if is_train_success_done:
                        self.training_steps.value += 1

            train_conditions = [
                self.total_time_steps.value >= self.next_train_time_step,
                self.parameter.AGENT_TYPE != AgentType.Reinforce
            ]
            if all(train_conditions):
                is_train_success_done = self.agent.train(
                    training_steps_v=self.training_steps.value
                )
                if is_train_success_done:
                    self.training_steps.value += 1

                self.next_train_time_step += self.parameter.TRAIN_INTERVAL_GLOBAL_TIME_STEPS

            if self.training_steps.value >= self.next_console_log:
                console_log(
                    self.train_start_time,
                    self.total_episodes.value,
                    self.total_time_steps.value,
                    self.last_mean_episode_reward.value,
                    self.n_rollout_transitions.value,
                    self.training_steps.value,
                    self.agent,
                    self.parameter
                )
                self.next_console_log += self.parameter.CONSOLE_LOG_INTERVAL_TRAINING_STEPS

            if self.training_steps.value >= self.next_test_training_step:
                self.testing()
                self.next_test_training_step += self.parameter.TEST_INTERVAL_TRAINING_STEPS
                if self.parameter.USE_WANDB:
                    wandb_log(self, wandb_obj, self.parameter)
                self.test_idx.value += 1

            if self.training_steps.value >= self.parameter.MAX_TRAINING_STEPS:
                print("[TRAIN TERMINATION] MAX_TRAINING_STEPS ({0}) REACHES!!!".format(
                    self.parameter.MAX_TRAINING_STEPS
                ))
                self.is_terminated.value = True

        total_training_time = time.time() - self.train_start_time
        formatted_total_training_time = time.strftime('%H:%M:%S', time.gmtime(total_training_time))
        print("Total Training Terminated: {}".format(formatted_total_training_time))
        print("Transition Rolling Rate: {0:.3f}/sec.".format(self.n_rollout_transitions.value / total_training_time))
        print("Training Rate: {0:.3f}/sec.".format(self.training_steps.value / total_training_time))
        if self.parameter.USE_WANDB:
            wandb_obj.join()

    def run(self):
        self.train_loop(parallel=True)

    def testing(self):
        print("*" * 120)
        self.test_episode_reward_avg.value, \
        self.test_episode_reward_std.value = \
            self.play_for_testing(self.parameter.N_TEST_EPISODES)

        elapsed_time = time.time() - self.train_start_time
        formatted_elapsed_time = time.strftime('%H:%M:%S', time.gmtime(elapsed_time))

        print("[Test: {0}, Training Step: {1:6,}] "
              "Episode Reward - Average: {2:.3f}, Standard Dev.: {3:.3f}, Elapsed Time: {4} ".format(
            self.test_idx.value + 1, self.training_steps.value,
            self.test_episode_reward_avg.value, self.test_episode_reward_std.value,
            formatted_elapsed_time
        ))

        termination_conditions = [
            self.test_episode_reward_avg.value > self.parameter.EPISODE_REWARD_AVG_SOLVED,
            self.test_episode_reward_std.value < self.parameter.EPISODE_REWARD_STD_SOLVED
        ]

        if all(termination_conditions):
            # Console 및 Wandb 로그를 위한 사항
            self.training_steps.value += 1

            print("Solved in {0:,} steps ({1:,} training steps)!".format(
                self.total_time_steps.value,
                self.training_steps.value
            ))
            model_save(
                model=self.agent.model,
                env_name=self.parameter.ENV_NAME,
                agent_type_name=self.parameter.AGENT_TYPE.name,
                test_episode_reward_avg=self.test_episode_reward_avg.value,
                test_episode_reward_std=self.test_episode_reward_std.value,
                parameter=self.parameter
            )
            print("[TRAIN TERMINATION] TERMINATION CONDITION REACHES!!!")
            self.is_terminated.value = True

        print("*" * 120)

    def play_for_testing(self, n_test_episodes):
        episode_reward_lst = []
        for i in range(n_test_episodes):
            episode_reward = 0  # cumulative_reward

            # Environment 초기화와 변수 초기화
            observation = self.test_env.reset()
            observation = np.expand_dims(observation, axis=0)

            while True:
                action = self.agent.get_action(observation, mode=AgentMode.TEST)

                # action을 통해서 next_state, reward, done, info를 받아온다
                next_observation, reward, done, _ = self.test_env.step(action[0])
                next_observation = np.expand_dims(next_observation, axis=0)

                episode_reward += reward  # episode_reward 를 산출하는 방법은 감가률 고려하지 않는 이 라인이 더 올바름.
                observation = next_observation

                if done:
                    break

            episode_reward_lst.append(episode_reward)

        return np.average(episode_reward_lst), np.std(episode_reward_lst)
