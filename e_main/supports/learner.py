import warnings

from gym.spaces import Box, Discrete

from a_configuration.a_base_config.c_models.recurrent_convolutional_models import ConfigRecurrentConvolutionalModel
from a_configuration.a_base_config.c_models.recurrent_linear_models import ConfigRecurrentLinearModel

warnings.filterwarnings('ignore')
warnings.simplefilter("ignore")

from collections import deque

import torch
import torch.multiprocessing as mp
import numpy as np
import time

from e_main.supports.actor import Actor
from g_utils.commons import model_save, console_log, wandb_log, get_wandb_obj, get_train_env, get_single_env, MeanBuffer
from g_utils.types import AgentType, AgentMode, Transition


class Learner(mp.Process):
    def __init__(self, agent, queue, config=None):
        super(Learner, self).__init__()
        self.agent = agent
        self.queue = queue
        self.config = config

        self.train_env = None
        self.test_env = None

        self.n_actors = self.config.N_ACTORS
        self.n_vectorized_envs = self.config.N_VECTORIZED_ENVS
        self.n_actor_terminations = 0

        self.episode_rewards = np.zeros(shape=(self.n_actors, self.n_vectorized_envs))
        self.episode_reward_buffer = MeanBuffer(self.config.N_EPISODES_FOR_MEAN_CALCULATION)

        self.total_time_step = mp.Value('i', 0)
        self.total_episodes = mp.Value('i', 0)
        self.training_step = mp.Value('i', 0)

        self.n_rollout_transitions = mp.Value('i', 0)

        self.train_start_time = None
        self.last_mean_episode_reward = mp.Value('d', 0.0)

        self.is_terminated = mp.Value('i', False)

        self.test_episode_reward_avg = mp.Value('d', 0.0)
        self.test_episode_reward_std = mp.Value('d', 0.0)

        self.next_train_time_step = config.TRAIN_INTERVAL_GLOBAL_TIME_STEPS
        self.next_test_training_step = config.TEST_INTERVAL_TRAINING_STEPS
        self.next_console_log = config.CONSOLE_LOG_INTERVAL_TRAINING_STEPS

        self.test_idx = mp.Value('i', 0)

        self.transition_rolling_rate = mp.Value('d', 0.0)
        self.train_step_rate = mp.Value('d', 0.0)

        if self.config.AGENT_TYPE == AgentType.MUZERO:
            self.transition_generator = self.generator_muzero()
            # TODO : history 저장 logic 구현
        else:
            if queue is None:  # Sequential
                self.transition_generator = self.generator_on_policy_transition()
                self.histories = []
                for _ in range(self.config.N_VECTORIZED_ENVS):
                    self.histories.append(deque(maxlen=self.config.N_STEP))

        self.is_recurrent_model = any([
            isinstance(self.config.MODEL_PARAMETER, ConfigRecurrentLinearModel),
            isinstance(self.config.MODEL_PARAMETER, ConfigRecurrentConvolutionalModel)
        ])

    def generator_on_policy_transition(self):
        observations = self.train_env.reset()

        if self.is_recurrent_model:
            self.agent.model.init_recurrent_hidden()
            observations = [(observations, self.agent.model.recurrent_hidden)]

        actor_time_step = 0

        while True:
            actor_time_step += 1
            actions = self.agent.get_action(observations)

            if isinstance(self.agent.action_space, Discrete):
                scaled_actions = actions
            elif isinstance(self.agent.action_space, Box):
                scaled_actions = actions * self.agent.action_scale + self.agent.action_bias
            else:
                raise ValueError()

            next_observations, rewards, dones, infos = self.train_env.step(scaled_actions)
            if self.is_recurrent_model:
                next_observations = [(next_observations, self.agent.model.recurrent_hidden)]

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

                if len(self.histories[env_id]) == self.config.N_STEP or done:
                    n_step_transition = Actor.get_n_step_transition(
                        history=self.histories[env_id], env_id=env_id,
                        actor_id=0, info=info, done=done, config=self.config
                    )
                    yield n_step_transition

            observations = next_observations

            if self.is_terminated.value:
                break

        yield None

    def generator_muzero(self):
        # TODO
        pass

    def train_loop(self, parallel=False):
        if not parallel:  # parallel인 경우 actor에서 train_env 생성/관리
            self.train_env = get_train_env(self.config)

        self.test_env = get_single_env(self.config)

        if self.config.USE_WANDB:
            wandb_obj = get_wandb_obj(self.config, self.agent)
        else:
            wandb_obj = None

        self.train_start_time = time.time()

        while True:
            if parallel:
                n_step_transition = self.queue.get()
            else:
                n_step_transition = next(self.transition_generator)

            self.total_time_step.value += 1

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

                if self.config.AGENT_TYPE == AgentType.REINFORCE:
                    count_training_steps = self.agent.train(training_steps_v=self.training_step.value)
                    self.training_step.value += count_training_steps

            train_conditions = [
                self.total_time_step.value >= self.next_train_time_step,
                self.config.AGENT_TYPE != AgentType.REINFORCE
            ]
            if all(train_conditions):
                count_training_steps = self.agent.train(training_steps_v=self.training_step.value)
                self.training_step.value += count_training_steps

                self.next_train_time_step += self.config.TRAIN_INTERVAL_GLOBAL_TIME_STEPS

            if self.training_step.value >= self.next_console_log:
                total_training_time = time.time() - self.train_start_time
                self.transition_rolling_rate.value = self.n_rollout_transitions.value / total_training_time
                self.train_step_rate.value = self.training_step.value / total_training_time

                console_log(
                    total_episodes_v=self.total_episodes.value,
                    total_time_steps_v=self.total_time_step.value,
                    last_mean_episode_reward_v=self.last_mean_episode_reward.value,
                    n_rollout_transitions_v=self.n_rollout_transitions.value,
                    transition_rolling_rate_v=self.transition_rolling_rate.value,
                    train_steps_v=self.training_step.value,
                    train_step_rate_v=self.train_step_rate.value,
                    agent=self.agent,
                    config=self.config
                )
                self.next_console_log += self.config.CONSOLE_LOG_INTERVAL_TRAINING_STEPS

            if self.training_step.value >= self.next_test_training_step:
                self.testing()
                self.next_test_training_step += self.config.TEST_INTERVAL_TRAINING_STEPS
                if self.config.USE_WANDB:
                    wandb_log(self, wandb_obj, self.config)
                self.test_idx.value += 1

            if self.training_step.value >= self.config.MAX_TRAINING_STEPS:
                print("[TRAIN TERMINATION] MAX_TRAINING_STEPS ({0:,}) REACHES!!!".format(
                    self.config.MAX_TRAINING_STEPS
                ))
                self.is_terminated.value = True

        total_training_time = time.time() - self.train_start_time
        formatted_total_training_time = time.strftime('%H:%M:%S', time.gmtime(total_training_time))
        print("Total Training Terminated: {}".format(formatted_total_training_time))
        print("Transition Rolling Rate: {0:.3f}/sec.".format(self.n_rollout_transitions.value / total_training_time))
        print("Training Rate: {0:.3f}/sec.".format(self.training_step.value / total_training_time))
        if self.config.USE_WANDB:
            wandb_obj.join()

    def run(self):
        self.train_loop(parallel=True)

    def testing(self):
        print("*" * 150)
        self.test_episode_reward_avg.value, self.test_episode_reward_std.value = \
            self.play_for_testing(self.config.N_TEST_EPISODES)

        elapsed_time = time.time() - self.train_start_time
        formatted_elapsed_time = time.strftime('%H:%M:%S', time.gmtime(elapsed_time))

        print("[Test: {0}, Training Step: {1:6,}] "
              "{2} Episodes Reward - Average: {3:.3f}, Standard Dev.: {4:.3f}, Elapsed Time from Training Start: {5} ".format(
            self.test_idx.value + 1, self.training_step.value, self.config.N_TEST_EPISODES,
            self.test_episode_reward_avg.value, self.test_episode_reward_std.value, formatted_elapsed_time
        ))

        termination_conditions = [
            self.test_episode_reward_avg.value > self.config.EPISODE_REWARD_AVG_SOLVED,
            self.test_episode_reward_std.value < self.config.EPISODE_REWARD_STD_SOLVED
        ]

        if all(termination_conditions):
            # Console 및 Wandb 로그를 위한 사항
            self.training_step.value += 1

            print("Solved in {0:,} steps ({1:,} training steps)!".format(
                self.total_time_step.value, self.training_step.value
            ))
            model_save(
                model=self.agent.model,
                env_name=self.config.ENV_NAME,
                agent_type_name=self.config.AGENT_TYPE.name,
                test_episode_reward_avg=self.test_episode_reward_avg.value,
                test_episode_reward_std=self.test_episode_reward_std.value,
                config=self.config
            )
            print("[TRAIN TERMINATION] TERMINATION CONDITION REACHES!!!")
            self.is_terminated.value = True

        print("*" * 150)

    def play_for_testing(self, n_test_episodes):
        self.agent.model.eval()

        episode_reward_lst = []

        for i in range(n_test_episodes):
            episode_reward = 0  # cumulative_reward

            # Environment 초기화와 변수 초기화
            observation = self.test_env.reset()
            observation = np.expand_dims(observation, axis=0)

            if self.is_recurrent_model:
                self.agent.model.init_recurrent_hidden()
                observation = [(observation, self.agent.model.recurrent_hidden)]

            while True:
                action = self.agent.get_action(observation, mode=AgentMode.TEST)
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
                            scaled_action = action * self.agent.action_scale[0] + self.agent.action_bias[0]
                        else:
                            scaled_action = action
                    elif action.ndim == 2:
                        if self.agent.action_scale is not None:
                            scaled_action = action[0] * self.agent.action_scale[0] + self.agent.action_bias[0]
                        else:
                            scaled_action = action[0]
                    else:
                        raise ValueError()
                else:
                    raise ValueError()

                next_observation, reward, done, _ = self.test_env.step(scaled_action)
                next_observation = np.expand_dims(next_observation, axis=0)

                if self.is_recurrent_model:
                    next_observation = [(next_observation, self.agent.model.recurrent_hidden)]

                episode_reward += reward  # episode_reward 를 산출하는 방법은 감가률 고려하지 않는 이 라인이 더 올바름.
                observation = next_observation

                if done:
                    break

            episode_reward_lst.append(episode_reward)

        self.agent.model.train()

        return np.average(episode_reward_lst), np.std(episode_reward_lst)
