# -*- coding:utf-8 -*-
import os
import pickle
import sys
import time
import zlib
from collections import deque
import numpy as np
import torch

idx = os.getcwd().index("link_rl")
PROJECT_HOME = os.getcwd()[:idx] + "link_rl"
if PROJECT_HOME not in sys.path:
    sys.path.append(PROJECT_HOME)

from common.fast_rl.common import utils
from common.fast_rl.rl_agent import float32_preprocessor
from common.fast_rl import actions, rl_agent, experience
from config.names import RLAlgorithmName, EnvironmentName
from rl_main.federated_main.utils import exp_moving_average
import rl_main.rl_utils as rl_utils
from config.parameters import PARAMETERS as params


class WorkerFastRL:
    def __init__(self, logger, worker_id, worker_mqtt_client, params):
        self.worker_id = worker_id
        self.worker_mqtt_client = worker_mqtt_client
        self.params = params

        self.env = rl_utils.get_environment(owner="worker", params=params)
        print("env:", params.ENVIRONMENT_ID)
        print("observation_space:", self.env.observation_space)
        print("action_space:", self.env.action_space)

        if params.ENVIRONMENT_ID in [EnvironmentName.PENDULUM_MATLAB_V0, EnvironmentName.PENDULUM_MATLAB_DOUBLE_AGENTS_V0]:
            self.env.start()

        self.rl_algorithm = rl_utils.get_rl_algorithm(env=self.env, worker_id=worker_id, logger=logger, params=params)

        self.episode_reward = 0

        self.global_max_ema_episode_reward = 0

        self.local_episode_rewards = []
        self.local_losses = []

        self.episode_reward_dequeue = deque(maxlen=self.params.AVG_EPISODE_SIZE_FOR_STAT)
        self.loss_dequeue = deque(maxlen=self.params.AVG_EPISODE_SIZE_FOR_STAT)

        self.episode_chief = -1

        self.is_success_or_fail_done = False
        self.logger = logger
        self.num_done_workers = 0

    def update_process(self, avg_gradients):
        self.rl_algorithm.model.set_gradients_to_current_parameters(avg_gradients)
        if self.params.RL_ALGORITHM in [RLAlgorithmName.DDPG_FAST_V0, RLAlgorithmName.D4PG_FAST_V0]:
            self.rl_algorithm.actor_optimizer.step()
            self.rl_algorithm.critic_optimizer.step()
        else:
            self.rl_algorithm.optimizer.step()

    def transfer_process(self, parameters):
        self.rl_algorithm.model.transfer_process(
            parameters, self.params.SOFT_TRANSFER, self.params.SOFT_TRANSFER_TAU
        )

    def send_msg(self, topic, msg):
        log_msg = "[SEND] TOPIC: {0}, PAYLOAD: 'episode': {1}, 'worker_id': {2} 'loss': {3}, 'episode_reward': {4} ".format(
            topic,
            msg['episode'],
            msg['worker_id'],
            msg['loss'],
            msg['episode_reward']
        )
        if self.params.MODE_PARAMETERS_TRANSFER and topic == self.params.MQTT_TOPIC_SUCCESS_DONE:
            log_msg += "'parameters_length': {0}".format(len(msg['parameters']))
        elif self.params.MODE_GRADIENTS_UPDATE and topic == self.params.MQTT_TOPIC_EPISODE_DETAIL:
            log_msg += "'gradients_length': {0}".format(len(msg['gradients']))
        elif topic == self.params.MQTT_TOPIC_FAIL_DONE:
            pass
        else:
            pass

        self.logger.info(log_msg)

        msg = pickle.dumps(msg, protocol=-1)
        msg = zlib.compress(msg)

        self.worker_mqtt_client.publish(topic=topic, payload=msg, qos=0, retain=False)

    def start_train(self):
        params.BATCH_SIZE *= params.TRAIN_STEP_FREQ

        if torch.cuda.is_available():
            device = torch.device("cuda" if params.CUDA else "cpu")
        else:
            device = torch.device("cpu")

        if params.RL_ALGORITHM in [RLAlgorithmName.DDPG_FAST_V0, RLAlgorithmName.D4PG_FAST_V0]:
            action_min = -self.params.ACTION_SCALE
            action_max = self.params.ACTION_SCALE

            if params.RL_ALGORITHM == RLAlgorithmName.DDPG_FAST_V0:
                action_selector = actions.DDPGActionSelector(
                    epsilon=params.EPSILON_INIT, ou_enabled=True, scale_factor=self.params.ACTION_SCALE
                )

                epsilon_tracker = actions.EpsilonTracker(
                    action_selector=action_selector,
                    eps_start=params.EPSILON_INIT,
                    eps_final=params.EPSILON_MIN,
                    eps_frames=params.EPSILON_MIN_STEP
                )

                agent = rl_agent.AgentDDPG(
                    self.rl_algorithm.model, n_actions=1, action_selector=action_selector,
                    action_min=action_min, action_max=action_max, device=device, preprocessor=float32_preprocessor
                )

            else:
                action_selector = actions.EpsilonGreedyD4PGActionSelector(
                    epsilon=params.EPSILON_INIT
                )

                epsilon_tracker = actions.EpsilonTracker(
                    action_selector=action_selector,
                    eps_start=params.EPSILON_INIT,
                    eps_final=params.EPSILON_MIN,
                    eps_frames=params.EPSILON_MIN_STEP
                )

                agent = rl_agent.AgentD4PG(
                    self.rl_algorithm.model, n_actions=1, action_selector=action_selector,
                    action_min=action_min, action_max=action_max, device=device, preprocessor=float32_preprocessor
                )

            experience_source = experience.ExperienceSourceSingleEnvFirstLast(
                self.env, agent, gamma=params.GAMMA, steps_count=params.N_STEP, step_length=-1
            )

            self.rl_algorithm.set_experience_source_to_buffer(experience_source=experience_source)

            step_idx = 0
            episode = 0

            with utils.RewardTracker(params=params, frame=None, stat=None, worker_id=self.worker_id) as reward_tracker:
                while step_idx < params.MAX_GLOBAL_STEPS:
                    # 1 스텝 진행하고 exp를 exp_queue에 넣음
                    step_idx += params.N_STEP

                    self.rl_algorithm.buffer.populate(params.TRAIN_STEP_FREQ)
                    epsilon_tracker.udpate(step_idx)

                    episode_rewards = experience_source.pop_episode_reward_lst()

                    if episode_rewards:
                        critic_loss_lst = []
                        actor_objective_lst = []

                        for _ in range(10):
                            gradients, critic_loss, actor_objective = self.rl_algorithm.train_net(step_idx=step_idx)
                            critic_loss_lst.append(critic_loss)
                            actor_objective_lst.append(actor_objective)

                        critic_loss = np.mean(critic_loss_lst)
                        actor_objective = np.mean(actor_objective)

                        current_episode_reward = episode_rewards[0]

                        solved, mean_episode_reward = reward_tracker.set_episode_reward(
                            current_episode_reward, step_idx, epsilon=action_selector.epsilon
                        )

                        if solved:
                            rl_agent.save_model(
                                os.path.join(PROJECT_HOME, "out", "model_save_files"),
                                params.ENVIRONMENT_ID.value,
                                self.rl_algorithm.model.__name__,
                                self.rl_algorithm.model,
                                step_idx,
                                mean_episode_reward
                            )

                        solved = self.interact_with_chief(
                            gradients, current_episode_reward, episode, step_idx, solved, critic_loss, actor_objective
                        )

                        if solved:
                            break

                        episode += 1

                if not solved:
                    self.interact_with_chief(
                        gradients, current_episode_reward, episode, step_idx, solved, critic_loss, actor_objective
                    )

    def interact_with_chief(self, gradients, episode_reward, episode, step_idx, solved, loss, actor_objective=None):
        self.local_losses.append(loss)
        self.local_episode_rewards.append(episode_reward)

        self.loss_dequeue.append(loss)
        self.episode_reward_dequeue.append(episode_reward)

        mean_episode_reward_over_recent_episodes = np.mean(self.episode_reward_dequeue)
        mean_loss_over_recent_episodes = np.mean(self.loss_dequeue)

        episode_msg = {
            "worker_id": self.worker_id,
            "episode": episode,
            "loss": loss,
            "episode_reward": episode_reward
        }

        if actor_objective:  # for Policy-Based RL
            episode_msg['actor_objective'] = actor_objective

        is_finish = False

        if solved:
            log_msg = "******* Worker {0} - Solved in episode {1}: Mean episode_reward = {2:8.3f}".format(
                self.worker_id,
                episode,
                mean_episode_reward_over_recent_episodes
            )
            self.logger.info(log_msg)
            print(log_msg)

            if self.params.MODE_GRADIENTS_UPDATE:
                episode_msg["gradients"] = gradients

            if self.params.MODE_PARAMETERS_TRANSFER:
                parameters = self.rl_algorithm.model.get_parameters()
                episode_msg["parameters"] = parameters

            self.send_msg(self.params.MQTT_TOPIC_SUCCESS_DONE, episode_msg)
            self.is_success_or_fail_done = True
            is_finish = True

        elif step_idx >= self.params.MAX_GLOBAL_STEPS:
            log_msg = "******* Worker {0} - Failed in episode {1}: Mean Episode Reward = {2}".format(
                self.worker_id,
                episode,
                mean_episode_reward_over_recent_episodes
            )
            self.logger.info(log_msg)
            print(log_msg)

            if self.params.MODE_GRADIENTS_UPDATE:
                episode_msg["gradients"] = gradients

            self.send_msg(self.params.MQTT_TOPIC_FAIL_DONE, episode_msg)
            self.is_success_or_fail_done = True
            is_finish = True

        else:
            ema_loss = exp_moving_average(self.local_losses, self.params.EMA_WINDOW)[-1]
            ema_episode_reward = exp_moving_average(self.local_episode_rewards, self.params.EMA_WINDOW)[-1]

            log_msg = "Worker {0}-Ep.{1:>2d}: Episode Reward={2:8.4f} (EMA: {3:>7.4f}, Mean: {4:>7.4f})".format(
                self.worker_id,
                episode,
                episode_reward,
                ema_episode_reward,
                mean_episode_reward_over_recent_episodes
            )

            log_msg += ", Loss={0:7.4f} (EMA: {1:7.4f}, Mean: {2:7.4f})".format(
                loss,
                ema_loss,
                mean_loss_over_recent_episodes
            )

            if self.params.EPSILON_GREEDY_ACT:
                log_msg += ", Epsilon: {0:5.2f}".format(
                    self.rl_algorithm.epsilon
                )

            self.logger.info(log_msg)
            #if self.params.VERBOSE: print(log_msg)

            if self.params.MODE_GRADIENTS_UPDATE:
                episode_msg["gradients"] = gradients

            self.send_msg(self.params.MQTT_TOPIC_EPISODE_DETAIL, episode_msg)

        while True:
            if episode == self.episode_chief:
                break
            time.sleep(0.01)

        return is_finish