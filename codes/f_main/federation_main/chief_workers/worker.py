# -*- coding:utf-8 -*-
import glob
import os
import pickle
import sys
import time
import zlib
from collections import deque
import numpy as np
import torch

current_path = os.path.dirname(os.path.realpath(__file__))
PROJECT_HOME = os.path.abspath(os.path.join(current_path, os.pardir, os.pardir, os.pardir, os.pardir))
if PROJECT_HOME not in sys.path:
    sys.path.append(PROJECT_HOME)

from codes.a_config.parameters import PARAMETERS as params

from codes.e_utils import rl_utils

from codes.f_main.federation_main.federated_main.utils import exp_moving_average

env = rl_utils.get_environment(owner="worker", params=params)


class Worker:
    def __init__(self, logger, worker_id, worker_mqtt_client, params):
        self.worker_id = worker_id
        self.worker_mqtt_client = worker_mqtt_client

        self.rl_algorithm = rl_utils.get_rl_algorithm(env=env, worker_id=worker_id, logger=logger, params=params)

        self.episode_reward = 0

        self.global_max_ema_episode_reward = 0

        self.local_episode_rewards = []
        self.local_losses = []

        self.episode_reward_dequeue = deque(maxlen=env.WIN_AND_LEARN_FINISH_CONTINUOUS_EPISODES)
        self.loss_dequeue = deque(maxlen=env.WIN_AND_LEARN_FINISH_CONTINUOUS_EPISODES)

        self.episode_chief = -1

        self.is_success_or_fail_done = False
        self.logger = logger
        self.params = params
        self.num_done_workers = 0

    def update_process(self, avg_gradients):
        self.rl_algorithm.model.set_gradients_to_current_parameters(avg_gradients)
        self.rl_algorithm.optimizer.step()

    def transfer_process(self, parameters):
        self.rl_algorithm.transfer_process(parameters, self.params.SOFT_TRANSFER, self.params.SOFT_TRANSFER_TAU)

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
        for episode in range(self.params.MAX_EPISODES):
            gradients, loss, episode_reward = self.rl_algorithm.on_episode(episode)
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

            if self.params.MODEL_SAVE:
                files = glob.glob(os.path.join(PROJECT_HOME, "out", "model_save_files", "{0}_*".format(self.worker_id)))
                for f in files:
                    os.remove(f)

                torch.save(
                    self.rl_algorithm.model.state_dict(),
                    os.path.join(
                        PROJECT_HOME, "out", "model_save_files",
                        "{0}_{1}_{2}_{3}.{4}.pt".format(
                            self.worker_id,
                            self.params.ENVIRONMENT_ID.name,
                            self.params.DEEP_LEARNING_MODEL.value,
                            self.params.RL_ALGORITHM.value,
                            episode
                        )
                    )
                )

            if mean_episode_reward_over_recent_episodes >= env.WIN_AND_LEARN_FINISH_SCORE and episode > env.WIN_AND_LEARN_FINISH_CONTINUOUS_EPISODES:
                log_msg = "******* Worker {0} - Solved in episode {1}: Mean episode_reward = {2:8.3f}".format(
                    self.worker_id,
                    episode,
                    mean_episode_reward_over_recent_episodes
                )
                self.logger.info(log_msg)
                print(log_msg)

                if self.params.MODE_PARAMETERS_TRANSFER:
                    parameters = self.rl_algorithm.get_parameters()
                    episode_msg["parameters"] = parameters

                self.send_msg(self.params.MQTT_TOPIC_SUCCESS_DONE, episode_msg)
                self.is_success_or_fail_done = True
                break

            elif episode == self.params.MAX_EPISODES - 1:
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
                break

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
                if self.params.VERBOSE: print(log_msg)

                if self.params.MODE_GRADIENTS_UPDATE:
                    episode_msg["gradients"] = gradients

                self.send_msg(self.params.MQTT_TOPIC_EPISODE_DETAIL, episode_msg)

            while True:
                if episode == self.episode_chief:
                    break
                time.sleep(0.01)
