# -*- coding:utf-8 -*-
import os
import pickle
import sys
import zlib

idx = os.getcwd().index("link_rl")
PROJECT_HOME = os.getcwd()[:idx] + "link_rl"
if PROJECT_HOME not in sys.path:
    sys.path.append(PROJECT_HOME)

from rl_main.federated_main.utils import exp_moving_average

import matplotlib.pyplot as plt
from matplotlib import gridspec
import csv

from collections import deque


class Chief:
    def __init__(self, logger, env, rl_model, params):
        self.logger = logger
        self.env = env

        self.messages_received_from_workers = {}

        self.NUM_DONE_WORKERS = 0
        self.episode_rewards = {}
        self.losses = {}

        self.episode_reward_over_recent_episodes = {}
        self.loss_over_recent_episodes = {}

        self.success_done_episode = {}
        self.success_done_episode_reward = {}

        self.global_max_ema_episode_reward = 0
        self.global_min_ema_loss = 1000000000

        self.episode_chief = 0
        self.num_messages = 0

        self.hidden_size = [params.HIDDEN_1_SIZE, params.HIDDEN_2_SIZE, params.HIDDEN_3_SIZE]

        self.model = rl_model

        self.params = params

        for worker_id in range(self.params.NUM_WORKERS):
            self.episode_rewards[worker_id] = []
            self.losses[worker_id] = []

            self.success_done_episode[worker_id] = []
            self.success_done_episode_reward[worker_id] = []

            self.episode_reward_over_recent_episodes[worker_id] = deque(maxlen=self.params.AVG_EPISODE_SIZE_FOR_STAT)
            self.loss_over_recent_episodes[worker_id] = deque(maxlen=self.params.AVG_EPISODE_SIZE_FOR_STAT)

    def update_loss_episode_reward(self, msg_payload):
        worker_id = msg_payload['worker_id']
        loss = msg_payload['loss']
        episode_reward = msg_payload['episode_reward']
        self.losses[worker_id].append(loss)
        self.episode_rewards[worker_id].append(episode_reward)
        self.loss_over_recent_episodes[worker_id].append(loss)
        self.episode_reward_over_recent_episodes[worker_id].append(episode_reward)

    def save_graph(self):
        plt.clf()

        fig = plt.figure(figsize=(30, 2 * self.params.NUM_WORKERS))
        gs = gridspec.GridSpec(
            nrows=self.params.NUM_WORKERS,  # row 몇 개
            ncols=2,  # col 몇 개
            width_ratios=[5, 5],
            hspace=0.2
        )

        max_episodes = 1
        for worker_id in range(self.params.NUM_WORKERS):
            if len(self.episode_rewards[worker_id]) > max_episodes:
                max_episodes = len(self.episode_rewards[worker_id])

        ax = {}
        for row in range(self.params.NUM_WORKERS):
            ax[row] = {}
            for col in range(2):
                ax[row][col] = plt.subplot(gs[row * 2 + col])
                ax[row][col].set_xlim([0, max_episodes])
                ax[row][col].tick_params(axis='both', which='major', labelsize=10)

        for worker_id in range(self.params.NUM_WORKERS):
            ax[worker_id][0].plot(
                range(len(self.losses[worker_id])),
                self.losses[worker_id],
                c='blue'
            )
            ax[worker_id][0].plot(
                range(len(self.losses[worker_id])),
                exp_moving_average(self.losses[worker_id], self.params.EMA_WINDOW),
                c='green'
            )

            ax[worker_id][1].plot(
                range(len(self.episode_rewards[worker_id])),
                self.episode_rewards[worker_id],
                c='blue'
            )
            ax[worker_id][1].plot(
                range(len(self.episode_rewards[worker_id])),
                exp_moving_average(self.episode_rewards[worker_id], self.params.EMA_WINDOW),
                c='green'
            )

            ax[worker_id][1].scatter(
                self.success_done_episode[worker_id],
                self.success_done_episode_reward[worker_id],
                marker="*",
                s=70,
                c='red'
            )

        plt.savefig(os.path.join(PROJECT_HOME, "out", "graphs", "loss_episode_reward.png"))
        plt.close('all')

    def save_results(self, worker_id, loss, ema_loss, episode_reward, ema_episode_reward):
        save_dir = os.path.join(PROJECT_HOME, "out", "save_results", "outputs.csv")
        f = open(save_dir, 'a', encoding='utf-8', newline='')
        wr = csv.writer(f)
        wr.writerow([self.episode_chief, worker_id, loss, ema_loss, episode_reward, ema_episode_reward])
        f.close()

    def process_message(self, topic, msg_payload):
        self.update_loss_episode_reward(msg_payload)
        self.save_graph()

        if topic == self.params.MQTT_TOPIC_EPISODE_DETAIL and self.params.MODE_GRADIENTS_UPDATE:
            self.model.accumulate_gradients(msg_payload['gradients'])
            # if msg_payload['episode'] == 0:
            #     self.model.accumulate_gradients(msg_payload['gradients'])
            # else:
            #     self.model.get_episode_reward_weighted_gradients(self.params.NUM_WORKERS - self.NUM_DONE_WORKERS,
            #                                         self.episode_reward_over_recent_episodes, msg_payload['gradients'],
            #                                         msg_payload['worker_id'], msg_payload['episode'])

        elif topic == self.params.MQTT_TOPIC_SUCCESS_DONE:
            self.success_done_episode[msg_payload['worker_id']].append(msg_payload['episode'])
            self.success_done_episode_reward[msg_payload['worker_id']].append(msg_payload['episode_reward'])

            self.NUM_DONE_WORKERS += 1
            print("CHIEF SUCCESS CHECK! - num_of_done_workers:", self.NUM_DONE_WORKERS)

        elif topic == self.params.MQTT_TOPIC_FAIL_DONE:
            self.NUM_DONE_WORKERS += 1
            print("CHIEF FAIL CHECK! - num_of_done_workers:", self.NUM_DONE_WORKERS)

        else:
            pass

    def get_transfer_ack_msg(self, parameters_transferred):
        log_msg = "[SEND] TOPIC: {0}, PAYLOAD: 'episode': {1}".format(
            self.params.MQTT_TOPIC_TRANSFER_ACK,
            self.episode_chief
        )

        transfer_msg = {
            "episode_chief": self.episode_chief
        }

        if self.params.MODE_PARAMETERS_TRANSFER:
            log_msg += ", 'parameters_length': {0}\n".format(
                len(parameters_transferred)
            )

            transfer_msg = {
                "episode_chief": self.episode_chief,
                "parameters": parameters_transferred
            }
        else:
            log_msg += ", No Transfer\n"

        self.logger.info(log_msg)

        transfer_msg = pickle.dumps(transfer_msg, protocol=-1)
        transfer_msg = zlib.compress(transfer_msg)

        if self.params.MODE_GRADIENTS_UPDATE:
            self.model.reset_average_gradients()
            self.model.reset_weighted_gradients()

        return transfer_msg

    def get_update_ack_msg(self, msg_payload):
        if self.params.MODE_GRADIENTS_UPDATE:
            log_msg = "[SEND] TOPIC: {0}, PAYLOAD: 'episode': {1}, 'global_avg_grad_length': {2}\n".format(
                self.params.MQTT_TOPIC_UPDATE_ACK,
                self.episode_chief,
                len(self.model.avg_gradients)
            )

            self.model.get_average_gradients(self.params.NUM_WORKERS - self.NUM_DONE_WORKERS)

            grad_update_msg = {
                "episode_chief": self.episode_chief,
                "avg_gradients": self.model.avg_gradients
            }
            ## weighted_gradients sharing
            # self.model.get_episode_reward_weighted_gradients(NUM_WORKERS - self.NUM_DONE_WORKERS, self.episode_reward_over_recent_episodes, msg_payload['gradients'], msg_payload['worker_id'])
            #
            # if msg_payload['episode'] == 0:
            #     self.model.get_average_gradients(self.params.NUM_WORKERS - self.NUM_DONE_WORKERS)
            #
            #     grad_update_msg = {
            #         "episode_chief": self.episode_chief,
            #         "avg_gradients": self.model.avg_gradients
            #     }
            #
            # else:
            #     grad_update_msg = {
            #         "episode_chief": self.episode_chief,
            #         "avg_gradients": self.model.weighted_gradients
            #     }
        else:
            log_msg = "[SEND] TOPIC: {0}, PAYLOAD: 'episode': {1}\n".format(
                self.params.MQTT_TOPIC_UPDATE_ACK,
                self.episode_chief
            )

            grad_update_msg = {
                "episode_chief": self.episode_chief
            }

        self.logger.info(log_msg)

        grad_update_msg = pickle.dumps(grad_update_msg, protocol=-1)
        grad_update_msg = zlib.compress(grad_update_msg)

        if self.params.MODE_GRADIENTS_UPDATE:
            self.model.reset_average_gradients()
            self.model.reset_weighted_gradients()

        return grad_update_msg
