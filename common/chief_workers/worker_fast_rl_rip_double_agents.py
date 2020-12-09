# -*- coding:utf-8 -*-
import os
import pickle
import sys
import time
from collections import deque
import numpy as np
import torch
import matplotlib.pyplot as plt
import zlib

idx = os.getcwd().index("link_rl")
PROJECT_HOME = os.getcwd()[:idx] + "link_rl"
if PROJECT_HOME not in sys.path:
    sys.path.append(PROJECT_HOME)

from rl_main import rl_utils
from rl_main.federated_main.utils import exp_moving_average
from common.fast_rl.rl_agent import float32_preprocessor
from rl_main.matlab_pendulum_main.experience_pendulum_ddpg_two_two_status import \
    ExperienceSourceSingleEnvFirstLastDdpgTwo, AgentType, RewardTrackerMatlabPendulum
from common.fast_rl import actions, rl_agent, experience
from config.names import RLAlgorithmName
from config.parameters import PARAMETERS as params


class WorkerFastRLRipDoubleAgents:
    def __init__(self, logger, worker_id, worker_mqtt_client, params):
        self.worker_id = worker_id
        self.worker_mqtt_client = worker_mqtt_client
        self.params = params

        self.env = rl_utils.get_environment(owner="worker", params=params)
        print("env:", params.ENVIRONMENT_ID)
        print("observation_space:", self.env.observation_space)
        print("action_space:", self.env.action_space)

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
        self.episode_reward_list = []
        self.num_done_workers = 0

    def swing_up_update_process(self, avg_gradients):
        self.rl_algorithm.swing_up_model.set_gradients_to_current_parameters(avg_gradients)
        self.rl_algorithm.swing_up_actor_optimizer.step()
        self.rl_algorithm.swing_up_critic_optimizer.step()

    def balancing_update_process(self, avg_gradients):
        self.rl_algorithm.balancing_model.set_gradients_to_current_parameters(avg_gradients)
        self.rl_algorithm.balancing_actor_optimizer.step()
        self.rl_algorithm.balancing_critic_optimizer.step()

    def swing_up_transfer_process(self, parameters):
        self.rl_algorithm.swing_up_model.transfer_process(
            parameters, self.params.SOFT_TRANSFER, self.params.SOFT_TRANSFER_TAU
        )

    def balancing_transfer_process(self, parameters):
        self.rl_algorithm.balancing_model.transfer_process(
            parameters, self.params.SOFT_TRANSFER, self.params.SOFT_TRANSFER_TAU
        )

    def send_msg(self, topic, msg, agent_type):
        log_msg = "[SEND] TOPIC: {0}, PAYLOAD: 'episode': {1}, 'worker_id': {2} 'loss': {3}, 'episode_reward': {4} ".format(
            topic,
            msg['episode'],
            msg['worker_id'],
            msg['loss'],
            msg['episode_reward']
        )

        log_msg += ", agent_type: {0}".format(agent_type.value)

        if self.params.MODE_PARAMETERS_TRANSFER and topic == self.params.MQTT_TOPIC_SUCCESS_DONE:
            log_msg += ", 'parameters_length': {0}".format(len(msg['parameters']))
        elif self.params.MODE_GRADIENTS_UPDATE and topic == self.params.MQTT_TOPIC_EPISODE_DETAIL:
            log_msg += ", 'gradients_length': {0}".format(len(msg['gradients']))
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

        if params.RL_ALGORITHM is RLAlgorithmName.DDPG_FAST_DOUBLE_AGENTS_V0:
            ######################
            ### SWING_UP Agent ###
            ######################
            swing_up_action_min = -self.params.SWING_UP_SCALE_FACTOR
            swing_up_action_max = self.params.SWING_UP_SCALE_FACTOR

            swing_up_action_selector = actions.DDPGActionSelector(
                epsilon=params.EPSILON_INIT, ou_enabled=True, scale_factor=self.params.SWING_UP_SCALE_FACTOR
            )

            swing_up_epsilon_tracker = actions.EpsilonTracker(
                action_selector=swing_up_action_selector,
                eps_start=self.params.EPSILON_INIT,
                eps_final=self.params.EPSILON_MIN,
                eps_frames=self.params.EPSILON_SWING_UP_MIN_STEP
            )

            swing_up_agent = rl_agent.AgentDDPG(
                self.rl_algorithm.swing_up_model, n_actions=1, action_selector=swing_up_action_selector,
                action_min=swing_up_action_min, action_max=swing_up_action_max,
                device=device, preprocessor=float32_preprocessor,
                name="SwingUp_AgentDDPG"
            )

            #######################
            ### BALANCING Agent ###
            #######################
            balancing_action_min = -self.params.BALANCING_SCALE_FACTOR
            balancing_action_max = self.params.BALANCING_SCALE_FACTOR

            balancing_action_selector = actions.DDPGActionSelector(
                epsilon=params.EPSILON_INIT, ou_enabled=True, scale_factor=self.params.BALANCING_SCALE_FACTOR
            )

            balancing_epsilon_tracker = actions.EpsilonTracker(
                action_selector=balancing_action_selector,
                eps_start=self.params.EPSILON_INIT,
                eps_final=self.params.EPSILON_MIN,
                eps_frames=self.params.EPSILON_BALANCING_MIN_STEP
            )

            balancing_agent = rl_agent.AgentDDPG(
                self.rl_algorithm.balancing_model, n_actions=1, action_selector=balancing_action_selector,
                action_min=balancing_action_min, action_max=balancing_action_max, device=device,
                preprocessor=float32_preprocessor,
                name="Balancing_AgentDDPG"
            )

            experience_source = ExperienceSourceSingleEnvFirstLastDdpgTwo(
                self.params, self.env, swing_up_agent, balancing_agent, gamma=self.params.GAMMA,
                steps_count=params.N_STEP, step_length=-1
            )

            self.rl_algorithm.set_buffer(experience_source=None)
            exp_source_iter = iter(experience_source)

            step_idx = 0
            episode = 0
            swing_up_step_idx = 0
            balancing_step_idx = 0
            agent_type = None

            with RewardTrackerMatlabPendulum(params=params, frame=None, stat=None, worker_id=self.worker_id) as reward_tracker:
                while step_idx < params.MAX_GLOBAL_STEPS:
                    # 1 스텝 진행하고 exp를 exp_queue에 넣음
                    step_idx += params.N_STEP

                    exp = next(exp_source_iter)

                    status_value = int(exp[0][-1])

                    if status_value == AgentType.SWING_UP_AGENT.value:  # SWING_UP: 0
                        swing_up_step_idx += 1
                        swing_up_epsilon_tracker.udpate(swing_up_step_idx)

                        self.rl_algorithm.swing_up_buffer._add(exp)
                        gradients, loss = self.rl_algorithm.train_swing_up_net(step_idx=step_idx)
                        agent_type = AgentType.SWING_UP_AGENT

                    elif status_value == AgentType.BALANCING_AGENT.value:  # BALANCING: 1
                        balancing_step_idx += 1
                        balancing_epsilon_tracker.udpate(balancing_step_idx)

                        self.rl_algorithm.balancing_buffer._add(exp)
                        gradients, loss = self.rl_algorithm.train_balancing_net(step_idx=step_idx)
                        agent_type = AgentType.BALANCING_AGENT

                    else:
                        raise ValueError()

                    episode_reward_and_info_lst = experience_source.pop_episode_reward_and_info_lst()

                    if episode_reward_and_info_lst:
                        current_episode_reward_and_info = episode_reward_and_info_lst[-1]
                        current_episode_reward = current_episode_reward_and_info[0]
                        self.episode_reward_list.append(current_episode_reward)

                        with open('episode_reward_list.txt', 'wb') as f:
                            pickle.dump(self.episode_reward_list, f)

                        solved, mean_episode_reward = reward_tracker.set_episode_reward(
                            current_episode_reward_and_info, step_idx,
                            epsilon=(swing_up_action_selector.epsilon, balancing_action_selector.epsilon)
                        )

                        if solved:
                            rl_agent.save_model(
                                os.path.join(PROJECT_HOME, "out", "model_save_files"),
                                params.ENVIRONMENT_ID.value,
                                self.rl_algorithm.swing_up_model.__name__,
                                self.rl_algorithm.swing_up_model,
                                step_idx,
                                mean_episode_reward
                            )

                            rl_agent.save_model(
                                os.path.join(PROJECT_HOME, "out", "model_save_files"),
                                params.ENVIRONMENT_ID,
                                self.rl_algorithm.balancing_model.__name__,
                                self.rl_algorithm.balancing_model,
                                step_idx,
                                mean_episode_reward
                            )

                            solved = self.interact_with_chief(
                                loss, gradients, current_episode_reward, episode, step_idx, solved, agent_type=agent_type
                            )

                        if solved:
                            break

                        episode += 1

                if not solved:
                    self.interact_with_chief(
                        loss, gradients, current_episode_reward, episode, step_idx, solved, agent_type=agent_type
                    )

            with open('episode_reward_list.txt', 'rb') as f:
                data = pickle.load(f)
            x = [i+1 for i in range(len(data))]
            y = data

            print(x)
            print(y)

            plt.plot(x,y)
            plt.title("MATLAB DDPG EPISODE REWARD")
            plt.xticks([i for i in range(params.MAX_GLOBAL_STEPS) if i%100000 ==0 ])
            plt.xlabel('episode')
            plt.ylabel('episode reward')
            plt.show()

    def interact_with_chief(self, loss, gradients, episode_reward, episode, step_idx, solved, agent_type):
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

        if agent_type:
            episode_msg['agent_type'] = agent_type.value

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

            self.send_msg(self.params.MQTT_TOPIC_SUCCESS_DONE, episode_msg, agent_type=agent_type)
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

            self.send_msg(self.params.MQTT_TOPIC_FAIL_DONE, episode_msg, agent_type=agent_type)
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

            self.send_msg(self.params.MQTT_TOPIC_EPISODE_DETAIL, episode_msg, agent_type=agent_type)

        while True:
            if episode == self.episode_chief:
                break
            time.sleep(0.01)

        return is_finish