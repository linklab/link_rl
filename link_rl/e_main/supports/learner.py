import warnings

import torch.multiprocessing as mp
import numpy as np
import time

from gym.spaces import Box, Discrete
from gym.vector import VectorEnv

from link_rl.e_main.supports.actor import Actor

warnings.filterwarnings('ignore')
warnings.simplefilter("ignore")

from link_rl.g_utils.commons import model_save, console_log, wandb_log, get_wandb_obj, get_train_env, get_single_env
from link_rl.g_utils.commons import MeanBuffer
from link_rl.g_utils.types import AgentType, AgentMode, OnPolicyAgentTypes, OffPolicyAgentTypes, HerConstant


class Learner(mp.Process):
    def __init__(self, agent, queue, shared_model_access_lock=None, config=None):
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

        self.train_start_time = None
        self.last_mean_episode_reward = mp.Value('d', 0.0)

        self.is_terminated = mp.Value('i', False)

        self.test_episode_reward_min = mp.Value('d', 0.0)

        self.test_episode_reward_best = 0.0

        self.next_train_time_step = config.TRAIN_INTERVAL_GLOBAL_TIME_STEPS
        self.next_test_training_step = config.TEST_INTERVAL_TRAINING_STEPS
        self.next_console_log = config.CONSOLE_LOG_INTERVAL_TRAINING_STEPS

        self.test_idx = mp.Value('i', 0)

        self.transition_rolling_rate = mp.Value('d', 0.0)
        self.train_step_rate = mp.Value('d', 0.0)

        self.single_actor_transition_generator = None

        if self.config.ACTION_MASKING:
            assert isinstance(self.agent.action_space, Discrete)

        self.shared_model_access_lock = shared_model_access_lock  # For only WorkingActor (A3C, AsynchronousPPO)

        self.modified_env_name = self.config.ENV_NAME.split("/")[
            1] if "/" in self.config.ENV_NAME else self.config.ENV_NAME

        self.single_actor = None  # for sequential (N_ACTOR == 1)

    def train_loop(self):
        if self.queue is None:
            self.single_actor = Actor(
                actor_id=0, agent=self.agent, queue=None, config=self.config
            )

            self.single_actor.set_train_env()

            if self.config.AGENT_TYPE == AgentType.TDMPC:
                self.single_actor_transition_generator = self.single_actor.generate_episode_for_single_env()
            elif self.config.N_VECTORIZED_ENVS == 1:
                self.single_actor_transition_generator = self.single_actor.generate_transition_for_single_env()
            else:
                self.single_actor_transition_generator = self.single_actor.generate_transition_for_vectorized_env()

        self.test_env = get_single_env(self.config, train_mode=False)

        if self.config.USE_WANDB:
            wandb_obj = get_wandb_obj(self.config, self.agent)
        else:
            wandb_obj = None

        self.train_start_time = time.time()

        while True:
            if self.config.AGENT_TYPE in [AgentType.A3C, AgentType.ASYNCHRONOUS_PPO]:
                async_info = self.queue.get()

                if async_info is None:
                    self.n_actor_terminations += 1
                    if self.n_actor_terminations >= self.n_actors:
                        self.is_terminated.value = True
                        break
                    else:
                        continue
                else:
                    if self.is_terminated.value:
                        continue

                if async_info["message_type"] == "DONE":
                    self.total_episodes.value += 1
                    self.episode_reward_buffer.add(async_info["episode_reward"])
                    self.last_mean_episode_reward.value = self.episode_reward_buffer.mean()
                elif async_info["message_type"] == "TRAIN":
                    self.training_step.value += async_info["count_training_steps"]
                    self.total_time_step.value += async_info["n_rollout_transitions"]
                else:
                    raise ValueError()

                last_train_env_info = async_info["last_train_env_info"]
            else:
                if self.queue is None:
                    n_step_transition = next(self.single_actor_transition_generator)
                else:
                    n_step_transition = self.queue.get()

                if n_step_transition is None:
                    self.n_actor_terminations += 1
                    if self.n_actor_terminations >= self.n_actors:
                        self.is_terminated.value = True
                        if self.queue is None:
                            self.single_actor.is_terminated.value = True
                        break
                    else:
                        continue
                else:
                    if self.is_terminated.value:
                        continue

                self.total_time_step.value += 1

                if self.config.AGENT_TYPE in OnPolicyAgentTypes:
                    self.agent.buffer.append(n_step_transition)
                elif self.config.AGENT_TYPE in OffPolicyAgentTypes:
                    self.agent.replay_buffer.append(n_step_transition)

                    if self.config.USE_HER:
                        self.agent.her_buffer.append(n_step_transition)
                else:
                    raise ValueError()

                last_train_env_info = n_step_transition.info

                if self.config.AGENT_TYPE == AgentType.TDMPC:
                    actor_id = n_step_transition.info["actor_id"]
                    env_id = n_step_transition.info["env_id"]
                    self.total_episodes.value += 1

                    self.episode_rewards[actor_id][env_id] = n_step_transition.cumulative_reward
                    self.episode_reward_buffer.add(self.episode_rewards[actor_id][env_id])
                    self.last_mean_episode_reward.value = self.episode_reward_buffer.mean()

                    self.episode_rewards[actor_id][env_id] = 0.0
                else:
                    actor_id = n_step_transition.info["actor_id"]
                    env_id = n_step_transition.info["env_id"]
                    self.episode_rewards[actor_id][env_id] += n_step_transition.reward
                    if n_step_transition.done:
                        self.total_episodes.value += 1
                        self.episode_reward_buffer.add(self.episode_rewards[actor_id][env_id])
                        self.last_mean_episode_reward.value = self.episode_reward_buffer.mean()

                        self.episode_rewards[actor_id][env_id] = 0.0

                        if self.config.USE_HER:
                            if n_step_transition.info[HerConstant.HER_SAVE_DONE]:
                                her_trajectory = self.agent.her_buffer.get_her_trajectory(
                                    n_step_transition.info[HerConstant.ACHIEVED_GOAL]
                                )
                                for her_transition in her_trajectory:
                                    self.agent.replay_buffer.append(her_transition)
                            self.agent.her_buffer.reset()

                ###################
                #   TRAIN START   #
                ###################
                reinforce_train_conditions = [
                    n_step_transition.done,
                    self.config.AGENT_TYPE == AgentType.REINFORCE
                ]
                train_conditions = [
                    self.total_time_step.value >= self.next_train_time_step,
                    self.config.AGENT_TYPE != AgentType.REINFORCE,
                ]
                if all(train_conditions) or all(reinforce_train_conditions):
                    count_training_steps = self.agent.train(training_steps_v=self.training_step.value)
                    self.training_step.value += count_training_steps
                    self.next_train_time_step += self.config.TRAIN_INTERVAL_GLOBAL_TIME_STEPS
                #################
                #   TRAIN END   #
                #################

            if self.config.CUSTOM_ENV_STAT is not None:
                self.config.CUSTOM_ENV_STAT.train_evaluate(last_train_env_info)

            if self.training_step.value >= self.next_console_log:
                total_training_time = time.time() - self.train_start_time
                self.transition_rolling_rate.value = self.total_time_step.value / total_training_time
                self.train_step_rate.value = self.training_step.value / total_training_time

                console_log(
                    self,
                    total_episodes_v=self.total_episodes.value,
                    last_mean_episode_reward_v=self.last_mean_episode_reward.value,
                    n_rollout_transitions_v=self.total_time_step.value,
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
                if self.queue is None:
                    self.single_actor.is_terminated.value = True

        total_training_time = time.time() - self.train_start_time
        formatted_total_training_time = time.strftime('%H:%M:%S', time.gmtime(total_training_time))
        print("Total Training Terminated: {}".format(formatted_total_training_time))
        print("Transition Rolling Rate: {0:.3f}/sec.".format(self.total_time_step.value / total_training_time))
        print("Training Rate: {0:.3f}/sec.".format(self.training_step.value / total_training_time))
        if self.config.USE_WANDB:
            wandb_obj.finish()

    def run(self):
        self.train_loop()

    def testing(self):
        print("*" * 150)

        self.test_episode_reward_min.value = self.play_for_testing(self.config.N_TEST_EPISODES)

        elapsed_time = time.time() - self.train_start_time
        formatted_elapsed_time = time.strftime('%H:%M:%S', time.gmtime(elapsed_time))

        test_str = "[Test: {0}, Training Step: {1:6,}] {2} Episodes Reward - Minimum: {3:.3f}".format(
            self.test_idx.value + 1,
            self.training_step.value,
            self.config.N_TEST_EPISODES,
            self.test_episode_reward_min.value
        )

        if self.config.CUSTOM_ENV_STAT is not None:
            test_str += ", " + self.config.CUSTOM_ENV_STAT.test_evaluation_str()

        test_str += ", Elapsed Time from Training Start: {0}".format(formatted_elapsed_time)

        print(test_str)

        # model_save_conditions
        if self.test_episode_reward_min.value > self.test_episode_reward_best:
            self.test_episode_reward_best = self.test_episode_reward_min.value
            model_save(
                agent=self.agent,
                env_name=self.modified_env_name,
                agent_type_name=self.config.AGENT_TYPE.name,
                test_episode_reward_min=self.test_episode_reward_min.value,
                config=self.config
            )
            print("[BEST TEST RESULT] MODEL SAVED!!!")

        # termination_conditions
        if self.test_episode_reward_min.value >= self.config.EPISODE_REWARD_MIN_SOLVED:
            # # Console ?? Wandb ?????? ???? ????
            # self.training_step.value += 1

            print("Solved in {0:,} steps ({1:,} training steps)!".format(
                self.total_time_step.value, self.training_step.value
            ))

            model_save(
                agent=self.agent,
                env_name=self.modified_env_name,
                agent_type_name=self.config.AGENT_TYPE.name,
                test_episode_reward_min=self.test_episode_reward_min.value,
                config=self.config
            )
            print("[TRAIN TERMINATION] TERMINATION CONDITION REACHES!!!")
            self.is_terminated.value = True
            if self.queue is None:
                self.single_actor.is_terminated.value = True

        print("*" * 150)

    def play_for_testing(self, n_test_episodes):
        if self.config.AGENT_TYPE in [AgentType.A3C, AgentType.ASYNCHRONOUS_PPO]:
            self.shared_model_access_lock.acquire()

        self.agent.model.eval()

        episode_reward_lst = []

        if self.config.CUSTOM_ENV_STAT is not None:
            self.config.CUSTOM_ENV_STAT.test_reset()

        for i in range(n_test_episodes):
            episode_reward = 0  # cumulative_reward
            episode_step = 0

            observation, info = self.test_env.reset(return_info=True)

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

            done = False

            while not done:
                if self.config.AGENT_TYPE == AgentType.TDMPC:
                    action = self.agent.get_action(
                        obs=observation, mode=AgentMode.TEST, step=self.training_step.value, t0=episode_step == 0
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

                if not self.config.AGENT_TYPE == AgentType.TDMPC:
                    if not isinstance(self.test_env, VectorEnv):
                        next_observation = np.expand_dims(next_observation, axis=0)

                if self.agent.is_recurrent_model:
                    next_observation = [(next_observation, self.agent.model.recurrent_hidden)]

                if self.config.ACTION_MASKING:
                    unavailable_actions = [info['unavailable_actions']]

                episode_reward += reward
                observation = next_observation

            episode_reward_lst.append(episode_reward)

            if self.config.CUSTOM_ENV_STAT is not None:
                self.config.CUSTOM_ENV_STAT.test_episode_done(info=info)

        if self.config.CUSTOM_ENV_STAT is not None:
            self.config.CUSTOM_ENV_STAT.test_evaluate()

        self.agent.model.train()

        if self.config.AGENT_TYPE in [AgentType.A3C, AgentType.ASYNCHRONOUS_PPO]:
            self.shared_model_access_lock.release()

        return min(episode_reward_lst)
