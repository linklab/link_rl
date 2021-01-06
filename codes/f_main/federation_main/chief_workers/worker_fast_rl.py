# -*- coding:utf-8 -*-
import os
import pickle
import sys
import time
import zlib
from collections import deque
import numpy as np
import torch

from codes.e_utils.experience_tracker import RewardTracker

current_path = os.path.dirname(os.path.realpath(__file__))
PROJECT_HOME = os.path.abspath(os.path.join(current_path, os.pardir, os.pardir, os.pardir, os.pardir))
if PROJECT_HOME not in sys.path:
    sys.path.append(PROJECT_HOME)

from codes.a_config.parameters import PARAMETERS as params

from codes.e_utils import rl_utils
from codes.e_utils.actions import EpsilonGreedySomeTimesBlowDQNActionSelector, \
    EpsilonGreedySomeTimesBlowDDPGActionSelector
from codes.e_utils.common_utils import save_model
from codes.e_utils.experience_single import ExperienceSourceSingleEnvFirstLast
from codes.e_utils.names import EnvironmentName, RLAlgorithmName
from codes.f_main.federation_main.federated_main.utils import exp_moving_average

MODEL_SAVE_DIR = os.path.join(PROJECT_HOME, "out", "model_save_files")
if not os.path.exists(MODEL_SAVE_DIR):
    os.makedirs(MODEL_SAVE_DIR)

if torch.cuda.is_available():
    device = torch.device("cuda" if params.CUDA else "cpu")
else:
    device = torch.device("cpu")


class WorkerFastRL:
    def __init__(self, logger, worker_id, worker_mqtt_client, params):
        self.worker_id = worker_id
        self.worker_mqtt_client = worker_mqtt_client
        self.params = params

        self.env = rl_utils.get_environment(owner="actual_worker", params=params)
        print("env:", params.ENVIRONMENT_ID)
        print("observation_space:", self.env.observation_space)
        print("action_space:", self.env.action_space)

        if params.ENVIRONMENT_ID in [EnvironmentName.PENDULUM_MATLAB_V0, EnvironmentName.PENDULUM_MATLAB_DOUBLE_RIP_V0]:
            self.env.start()

        self.agent, self.epsilon_tracker = rl_utils.get_rl_agent(
            env=self.env, worker_id=0, params=params, device=device
        )

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
        self.agent.model.set_gradients_to_current_parameters(avg_gradients)

        if self.params.RL_ALGORITHM in [RLAlgorithmName.DDPG_FAST_V0, RLAlgorithmName.D4PG_FAST_V0]:
            self.agent.actor_optimizer.step()
            self.agent.critic_optimizer.step()
        else:
            self.agent.optimizer.step()

    def transfer_process(self, parameters):
        self.agent.model.transfer_process(
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

        if params.RL_ALGORITHM in [RLAlgorithmName.DQN_FAST_V0]:
            if params.ENVIRONMENT_ID in [EnvironmentName.PENDULUM_MATLAB_V0]:
                action_selector = EpsilonGreedySomeTimesBlowDQNActionSelector(
                    epsilon=params.EPSILON_INIT,
                    blowing_action_rate=0.0002,  # 5000 스텝에 1번 정도(지수 분포)의 주기로 Blowing Action 가해짐
                    min_blowing_action_idx=0,
                    max_blowing_action_idx=self.env.n_actions - 1,
                )
                self.agent.action_selector = action_selector
                self.epsilon_tracker.action_selector = action_selector
        elif params.RL_ALGORITHM in (RLAlgorithmName.DDPG_FAST_V0, RLAlgorithmName.D4PG_FAST_V0):
            if params.ENVIRONMENT_ID in [EnvironmentName.PENDULUM_MATLAB_V0]:
                action_selector = EpsilonGreedySomeTimesBlowDDPGActionSelector(
                    epsilon=params.EPSILON_INIT, ou_enabled=True, scale_factor=self.params.ACTION_SCALE,
                    blowing_action_rate=0.0002,  # 5000 스텝에 1번 정도(지수 분포)의 주기로 Blowing Action 가해짐
                    min_blowing_action=-10.0 * self.params.ACTION_SCALE,
                    max_blowing_action=10.0 * self.params.ACTION_SCALE,
                )
                self.agent.action_selector = action_selector
                self.epsilon_tracker.action_selector = action_selector
        else:
            raise ValueError()

        experience_source = ExperienceSourceSingleEnvFirstLast(
            self.env, self.agent, gamma=params.GAMMA, steps_count=params.N_STEP, step_length=-1
        )

        self.agent.set_experience_source_to_buffer(experience_source=experience_source)

        step_idx = 0
        last_loss = 0.0
        episode = 0
        stat = None

        with RewardTracker(params=params, frame=None, stat=stat, worker_id=self.worker_id) as reward_tracker:
            while step_idx < params.MAX_GLOBAL_STEP:
                # 1 스텝 진행하고 exp를 exp_queue에 넣음
                step_idx += params.TRAIN_STEP_FREQ
                last_entry = self.agent.buffer.populate(params.TRAIN_STEP_FREQ)
                self.epsilon_tracker.udpate(step_idx)

                ###################  TRAIN!!!
                actor_objective = None

                if self.params.RL_ALGORITHM == RLAlgorithmName.DQN_FAST_V0:
                    gradients, loss = self.agent.train_net(step_idx=step_idx)
                elif self.params.RL_ALGORITHM in [RLAlgorithmName.DDPG_FAST_V0, RLAlgorithmName.D4PG_FAST_V0]:
                    gradients, loss, actor_objective = self.agent.train_net(step_idx=step_idx)
                else:
                    raise ValueError()
                ###################

                episode_rewards = experience_source.pop_episode_reward_lst()

                if episode_rewards:
                    for current_episode_reward in episode_rewards:
                        solved, mean_episode_reward = reward_tracker.set_episode_reward(
                            current_episode_reward, step_idx, epsilon=self.agent.action_selector.epsilon,
                            last_info=last_entry.info, last_loss=last_loss, model=self.agent.model
                        )

                        if solved:
                            save_model(
                                MODEL_SAVE_DIR, params.ENVIRONMENT_ID.value, self.agent.model,
                                step_idx, mean_episode_reward
                            )

                        solved = self.interact_with_chief(
                            gradients, current_episode_reward, episode, step_idx, solved, loss, actor_objective
                        )

                        if solved:
                            break

                        episode += 1

            if not solved:
                self.interact_with_chief(
                    gradients, current_episode_reward, episode, step_idx, solved, loss, actor_objective
                )

            if params.SAVE_AT_MAX_GLOBAL_STEPS:
                save_model(
                    MODEL_SAVE_DIR, params.ENVIRONMENT_ID.value, self.agent.model,
                    step_idx, mean_episode_reward
                )

    def train_at_episode_end(self, step_idx):
        ###################
        loss_lst = []  # for actor critic model, loss means critic_loss
        actor_objective_lst = []
        actor_objective = None

        for _ in range(10):
            if self.params.RL_ALGORITHM == RLAlgorithmName.DQN_FAST_V0:
                gradients, loss = self.agent.train_net(step_idx=step_idx)
                loss_lst.append(loss)
            elif self.params.RL_ALGORITHM in [RLAlgorithmName.DDPG_FAST_V0, RLAlgorithmName.D4PG_FAST_V0]:
                gradients, loss, actor_objective = self.agent.train_net(step_idx=step_idx)
                loss_lst.append(loss)
                actor_objective_lst.append(actor_objective)
            else:
                raise ValueError()

        loss = np.mean(loss_lst)

        if self.params.RL_ALGORITHM in [RLAlgorithmName.DDPG_FAST_V0, RLAlgorithmName.D4PG_FAST_V0]:
            actor_objective = np.mean(actor_objective_lst)

        return loss, actor_objective
        ###################

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
                parameters = self.agent.model.get_parameters()
                episode_msg["parameters"] = parameters

            self.send_msg(self.params.MQTT_TOPIC_SUCCESS_DONE, episode_msg)
            self.is_success_or_fail_done = True
            is_finish = True

        elif step_idx >= self.params.MAX_GLOBAL_STEP:
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
                    self.agent.epsilon
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