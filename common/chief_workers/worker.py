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

idx = os.getcwd().index("{0}link_rl".format(os.sep))
PROJECT_HOME = os.getcwd()[:idx+1] + "link_rl{0}".format(os.sep)
sys.path.append(PROJECT_HOME)

from config.parameters_general import PARAMETERS_GENERAL as params
from rl_main.utils import exp_moving_average
import rl_main.rl_utils as rl_utils

env = rl_utils.get_environment(owner="worker")


class Worker:
    def __init__(self, logger, worker_id, worker_mqtt_client):
        self.worker_id = worker_id
        self.worker_mqtt_client = worker_mqtt_client

        self.rl_algorithm = rl_utils.get_rl_algorithm(env=env, worker_id=worker_id, logger=logger)

        self.score = 0

        self.global_max_ema_score = 0
        self.global_min_ema_loss = 1000000000

        self.local_scores = []
        self.local_losses = []

        self.score_dequeue = deque(maxlen=env.WIN_AND_LEARN_FINISH_CONTINUOUS_EPISODES)
        self.loss_dequeue = deque(maxlen=env.WIN_AND_LEARN_FINISH_CONTINUOUS_EPISODES)

        self.episode_chief = -1

        self.is_success_or_fail_done = False
        self.logger = logger

    def update_process(self, avg_gradients):
        self.rl_algorithm.model.set_gradients_to_current_parameters(avg_gradients)
        self.rl_algorithm.optimizer.step()

    def transfer_process(self, parameters):
        self.rl_algorithm.transfer_process(parameters, params.SOFT_TRANSFER, params.SOFT_TRANSFER_TAU)

    def send_msg(self, topic, msg):
        log_msg = "[SEND] TOPIC: {0}, PAYLOAD: 'episode': {1}, 'worker_id': {2} 'loss': {3}, 'score': {4} ".format(
            topic,
            msg['episode'],
            msg['worker_id'],
            msg['loss'],
            msg['score']
        )
        if params.MODE_PARAMETERS_TRANSFER and topic == params.MQTT_TOPIC_SUCCESS_DONE:
            log_msg += "'parameters_length': {0}".format(len(msg['parameters']))
        elif params.MODE_GRADIENTS_UPDATE and topic == params.MQTT_TOPIC_EPISODE_DETAIL:
            log_msg += "'gradients_length': {0}".format(len(msg['gradients']))
        elif topic == params.MQTT_TOPIC_FAIL_DONE:
            pass
        else:
            pass

        self.logger.info(log_msg)

        msg = pickle.dumps(msg, protocol=-1)
        msg = zlib.compress(msg)

        self.worker_mqtt_client.publish(topic=topic, payload=msg, qos=0, retain=False)

    def start_train(self):
        for episode in range(params.MAX_EPISODES):
            gradients, loss, score = self.rl_algorithm.on_episode(episode)
            self.local_losses.append(loss)
            self.local_scores.append(score)

            self.loss_dequeue.append(loss)
            self.score_dequeue.append(score)

            mean_score_over_recent_100_episodes = np.mean(self.score_dequeue)
            mean_loss_over_recent_100_episodes = np.mean(self.loss_dequeue)

            episode_msg = {
                "worker_id": self.worker_id,
                "episode": episode,
                "loss": loss,
                "score": score
            }

            if params.MODEL_SAVE:
                files = glob.glob(os.path.join(PROJECT_HOME, "model_save_files", "{0}_*".format(self.worker_id)))
                for f in files:
                    os.remove(f)

                torch.save(
                    self.rl_algorithm.model.state_dict(),
                    os.path.join(
                        PROJECT_HOME, "model_save_files",
                        "{0}_{1}_{2}_{3}.{4}.pt".format(
                            self.worker_id,
                            params.ENVIRONMENT_ID.name,
                            params.DEEP_LEARNING_MODEL.value,
                            params.RL_ALGORITHM.value,
                            episode
                        )
                    )
                )

            if mean_score_over_recent_100_episodes >= env.WIN_AND_LEARN_FINISH_SCORE and episode > env.WIN_AND_LEARN_FINISH_CONTINUOUS_EPISODES:
                log_msg = "******* Worker {0} - Solved in episode {1}: Mean score = {2}".format(
                    self.worker_id,
                    episode,
                    mean_score_over_recent_100_episodes
                )
                self.logger.info(log_msg)
                print(log_msg)

                if params.MODE_PARAMETERS_TRANSFER:
                    parameters = self.rl_algorithm.get_parameters()
                    episode_msg["parameters"] = parameters

                self.send_msg(params.MQTT_TOPIC_SUCCESS_DONE, episode_msg)
                self.is_success_or_fail_done = True
                break

            elif episode == params.MAX_EPISODES - 1:
                log_msg = "******* Worker {0} - Failed in episode {1}: Mean score = {2}".format(
                    self.worker_id,
                    episode,
                    mean_score_over_recent_100_episodes
                )
                self.logger.info(log_msg)
                print(log_msg)

                if params.MODE_GRADIENTS_UPDATE:
                    episode_msg["gradients"] = gradients

                self.send_msg(params.MQTT_TOPIC_FAIL_DONE, episode_msg)
                self.is_success_or_fail_done = True
                break

            else:
                ema_loss = exp_moving_average(self.local_losses, params.EMA_WINDOW)[-1]
                ema_score = exp_moving_average(self.local_scores, params.EMA_WINDOW)[-1]

                log_msg = "Worker {0}-Ep.{1:>2d}: Loss={2:6.4f} (EMA: {3:6.4f}, Mean: {4:6.4f})".format(
                    self.worker_id,
                    episode,
                    loss,
                    ema_loss,
                    mean_loss_over_recent_100_episodes
                )

                log_msg += ", Score={0:6.4f} (EMA: {1:>6.4f}, Mean: {2:>6.4f})".format(
                    score,
                    ema_score,
                    mean_score_over_recent_100_episodes
                )

                if params.EPSILON_GREEDY_ACT:
                    log_msg += ", Epsilon: {0:5.2f}".format(
                        self.rl_algorithm.epsilon
                    )

                self.logger.info(log_msg)
                if params.VERBOSE: print(log_msg)

                if params.MODE_GRADIENTS_UPDATE:
                    episode_msg["gradients"] = gradients

                self.send_msg(params.MQTT_TOPIC_EPISODE_DETAIL, episode_msg)

            while True:
                if episode == self.episode_chief:
                    break
                time.sleep(0.01)
