from collections import deque

import torch
import torch.multiprocessing as mp
import numpy as np
import time

from e_main.supports.actor import Actor
from g_utils.commons import model_save, console_log, wandb_log, get_wandb_obj, get_train_env, get_single_env
from g_utils.buffers import Buffer
from g_utils.types import AgentType, AgentMode, Transition


class Learner(mp.Process):
    def __init__(self, agent, queue, device=torch.device("cpu"), params=None):
        super(Learner, self).__init__()
        self.agent = agent
        self.queue = queue
        self.device = device
        self.params = params

        self.train_env = None
        self.test_env = None

        self.n_actors = self.params.N_ACTORS
        self.n_vectorized_envs = self.params.N_VECTORIZED_ENVS
        self.n_actor_terminations = 0

        self.episode_rewards = np.zeros((self.n_actors * self.n_vectorized_envs,))
        self.episode_reward_lst = []

        self.total_time_steps = mp.Value('i', 0)
        self.total_episodes = mp.Value('i', 0)
        self.training_steps = mp.Value('i', 0)

        self.buffer = Buffer(
            capacity=params.BUFFER_CAPACITY, device=self.device
        )

        self.n_rollout_transitions = mp.Value('i', 0)

        self.total_train_start_time = None
        self.last_mean_episode_reward = mp.Value('d', 0.0)

        self.is_terminated = mp.Value('i', False)

        self.test_episode_reward_avg = mp.Value('d', 0.0)
        self.test_episode_reward_std = mp.Value('d', 0.0)

        self.next_train_time_step = params.TRAIN_INTERVAL_TOTAL_TIME_STEPS
        self.next_test_training_step = params.TEST_INTERVAL_TRAINING_STEPS
        self.next_console_log = params.CONSOLE_LOG_INTERVAL_TOTAL_TIME_STEPS

        if queue is None: # SYNC
            self.transition_generator = self.generator_on_policy_transition()

            self.histories = []
            for _ in range(self.params.N_VECTORIZED_ENVS):
                self.histories.append(deque(maxlen=self.params.N_STEP))

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

                if len(self.histories[env_id]) == self.params.N_STEP or done:
                    n_step_transition = Actor.get_n_step_transition(
                        history=self.histories[env_id], env_id=env_id,
                        actor_id=0, info=info, done=done, params=self.params
                    )
                    yield n_step_transition

            observations = next_observations

            if self.is_terminated.value:
                break

        yield None

    def train_loop(self, sync=True):
        if sync:
            self.train_env = get_train_env(self.params)

        self.test_env = get_single_env(self.params)

        if self.params.USE_WANDB:
            wandb_obj = get_wandb_obj(self.params)
        else:
            wandb_obj = None

        self.total_train_start_time = time.time()

        while True:
            if sync:
                n_step_transition = next(self.transition_generator)
            else:
                n_step_transition = self.queue.get()

            # print(n_step_transition.info["training_step_v"], "&", end=' ')

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
                else:
                    self.total_time_steps.value += 1

            self.buffer.append(n_step_transition)
            self.n_rollout_transitions.value += 1

            actor_id = n_step_transition.info["actor_id"]
            env_id = n_step_transition.info["env_id"]
            self.episode_rewards[actor_id * env_id] += n_step_transition.reward

            if self.total_time_steps.value >= self.next_train_time_step:
                if self.params.AGENT_TYPE != AgentType.Reinforce:
                    self.agent.train(
                        buffer=self.buffer,
                        training_steps=self.training_steps
                    )
                self.next_train_time_step += self.params.TRAIN_INTERVAL_TOTAL_TIME_STEPS

            if n_step_transition.done:
                self.total_episodes.value += 1

                self.episode_reward_lst.append(self.episode_rewards[actor_id * env_id])
                self.last_mean_episode_reward.value = np.mean(
                    self.episode_reward_lst[-1 * self.params.N_EPISODES_FOR_MEAN_CALCULATION:]
                )

                self.episode_rewards[actor_id * env_id] = 0.0

                if self.params.AGENT_TYPE == AgentType.Reinforce:
                    self.agent.train(
                        buffer=self.buffer,
                        training_steps=self.training_steps
                    )

            if self.total_time_steps.value >= self.next_console_log:
                console_log(
                    self.total_train_start_time, self.total_episodes.value,
                    self.total_time_steps.value,
                    self.last_mean_episode_reward.value,
                    self.n_rollout_transitions.value, self.training_steps.value,
                    self.agent, self.params
                )
                self.next_console_log += self.params.CONSOLE_LOG_INTERVAL_TOTAL_TIME_STEPS

            if self.training_steps.value >= self.next_test_training_step:
                self.testing()
                self.next_test_training_step += self.params.TEST_INTERVAL_TRAINING_STEPS
                if self.params.USE_WANDB:
                    wandb_log(self, wandb_obj, self.params)

            if self.training_steps.value >= self.params.MAX_TRAINING_STEPS:
                print("[TRAIN TERMINATION] MAX_TRAINING_STEPS ({0}) REACHES!!!".format(
                    self.params.MAX_TRAINING_STEPS
                ))
                self.is_terminated.value = True

        total_training_time = time.time() - self.total_train_start_time
        formatted_total_training_time = time.strftime(
            '%H:%M:%S', time.gmtime(total_training_time)
        )
        print("Total Training Terminated : {}".format(formatted_total_training_time))
        print("Transition Rolling Rate: {0:.3f}/sec.".format(
            self.n_rollout_transitions.value / total_training_time
        ))
        print("Training Rate: {0:.3f}/sec.".format(
            self.training_steps.value / total_training_time
        ))
        if self.params.USE_WANDB:
            wandb_obj.join()

    def run(self):
        self.train_loop(sync=False)

    def testing(self):
        print("*" * 80)
        self.test_episode_reward_avg.value, \
        self.test_episode_reward_std.value = \
            self.play_for_testing(self.params.N_TEST_EPISODES)

        print("[Test Episode Reward] Average: {0:.3f}, Standard Dev.: {1:.3f}".format(
            self.test_episode_reward_avg.value,
            self.test_episode_reward_std.value
        ))

        termination_conditions = [
            self.test_episode_reward_avg.value > self.params.EPISODE_REWARD_AVG_SOLVED,
            self.test_episode_reward_std.value < self.params.EPISODE_REWARD_STD_SOLVED
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
                env_name=self.params.ENV_NAME,
                agent_type_name=self.params.AGENT_TYPE.name,
                test_episode_reward_avg=self.test_episode_reward_avg.value,
                test_episode_reward_std=self.test_episode_reward_std.value
            )
            print("[TRAIN TERMINATION] TERMINATION CONDITION REACHES!!!")
            self.is_terminated.value = True

        print("*" * 80)

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
