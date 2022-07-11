import time
import warnings

import numpy as np
from gym.spaces import Discrete, Box
from gym.wrappers import LazyFrames

from link_rl.a_configuration.a_base_config.a_environments.competition_olympics import ConfigCompetitionOlympics
from link_rl.h_utils.commons_rl import Episode

warnings.filterwarnings('ignore')
warnings.simplefilter("ignore")

from collections import deque
import torch.multiprocessing as mp

from link_rl.h_utils.commons import get_train_env, get_single_env
from link_rl.h_utils.types import Transition, AgentType


class Actor(mp.Process):
    def __init__(self, actor_id, agent, queue, config, working_actor=False):
        super(Actor, self).__init__()
        self.actor_id = actor_id
        self.agent = agent
        self.queue = queue
        self.config = config

        self.is_env_created = mp.Value('i', False)
        self.is_terminated = mp.Value('i', False)

        self.train_env = None

        self.histories = []
        for _ in range(self.config.N_VECTORIZED_ENVS):
            self.histories.append(deque(maxlen=self.config.N_STEP))

        self.total_time_step = 0

        self.working_actor = working_actor
        if self.working_actor:
            assert self.config.AGENT_TYPE in [AgentType.A3C, AgentType.ASYNCHRONOUS_PPO]
            # FOR WORKER TRAIN
            self.training_step = 0
            self.next_train_time_step = config.TRAIN_INTERVAL_GLOBAL_TIME_STEPS
            self.episode_rewards = np.zeros(shape=(self.config.N_VECTORIZED_ENVS,))
            self.last_train_env_info = None

    def run(self):          # b_single_main_parallel.py로 실행하였을 때에만 자동으로 호출 (멀티프로세스 프레임워크)
        assert self.queue is not None

        self.set_train_env()

        self.agent.model.eval()
        
        try:
            if self.config.AGENT_TYPE == AgentType.TDMPC:
                next(self.generate_episode_for_single_env())
            elif self.config.N_VECTORIZED_ENVS == 1:
                next(self.generate_transition_for_single_env())
            else:
                next(self.generate_transition_for_vectorized_env())
        except StopIteration as e:
            pass

    def set_train_env(self):
        if self.config.AGENT_TYPE == AgentType.TDMPC or self.config.N_VECTORIZED_ENVS == 1:
            if isinstance(self.config, ConfigCompetitionOlympics):
                self.train_env = get_single_env(self.config, train_mode=True, agent=self.agent)
            else:
                self.train_env = get_single_env(self.config, train_mode=True)
        else:
            self.train_env = get_train_env(self.config)
        self.is_env_created.value = True

    def generate_transition_for_single_env(self):
        observation, info = self.train_env.reset(return_info=True)

        if self.agent.is_recurrent_model:
            self.agent.model.init_recurrent_hidden()

        while True:
            observations = np.expand_dims(observation, axis=0)

            if self.agent.is_recurrent_model:
                observations = [(observations, self.agent.model.recurrent_hidden)]

            if self.config.ACTION_MASKING:
                unavailable_actions = []
                unavailable_actions.append(info["unavailable_actions"])
            else:
                unavailable_actions = None

            action, scaled_action = self._get_action(observations, unavailable_actions, vectorized_env=False)

            next_observation, reward, done, info = self.train_env.step(scaled_action)

            self.total_time_step += 1

            info["actor_id"] = self.actor_id
            info["env_id"] = 0

            self.histories[0].append(Transition(
                observation=observation,
                action=action,
                next_observation=next_observation,
                reward=reward,
                done=done,
                info=info
            ))

            if self.working_actor:
                self.last_train_env_info = info

            if len(self.histories[0]) == self.config.N_STEP or done:
                n_step_transition, real_n_steps = Actor.get_n_step_transition(
                    history=self.histories[0], env_id=0, actor_id=self.actor_id,
                    info=info, done=done, total_time_step=self.total_time_step, config=self.config
                )

                if self.working_actor:
                    self.agent.buffer.append(n_step_transition)
                    self.episode_rewards[0] += n_step_transition.reward

                    self.queue.put({
                        "message_type": "TRANSITION",
                        "done": done,
                        "episode_reward": self.episode_rewards[0],
                        "last_train_env_info": self.last_train_env_info,
                        "real_n_steps": real_n_steps
                    })
                    if done:
                        self.episode_rewards[0] = 0.0
                else:
                    if self.queue is not None:
                        self.queue.put(n_step_transition)
                    else:
                        yield n_step_transition

            if done:
                next_observation, info = self.train_env.reset(return_info=True)

            observation = next_observation

            if self.working_actor and len(self.agent.buffer) >= self.config.BATCH_SIZE:
                self.working_train()

            if self.is_terminated.value:
                break

        if self.queue is not None:
            self.queue.put(None)
        else:
            yield None

    def generate_episode_for_single_env(self):
        step = 0
        while True:
            # Collect trajectory
            obs = self.train_env.reset()

            if type(obs) == LazyFrames:
                obs = np.asarray(obs)

            episode = Episode(self.config, obs, self.agent.n_out_actions)
            while not episode.done:
                action = self.agent.get_action(obs, step=step, t0=episode.first)
                obs, reward, done, info = self.train_env.step(action.cpu().numpy())

                if type(obs) == LazyFrames:
                    obs = np.asarray(obs)

                self.total_time_step += 1
                info["actor_id"] = self.actor_id
                info["env_id"] = 0
                episode += (obs, action, reward, done, info)

            assert len(episode) == int(self.config.FIXED_TOTAL_TIME_STEPS_PER_EPISODE / self.config.ACTION_REPEAT)
            step += int(self.config.FIXED_TOTAL_TIME_STEPS_PER_EPISODE / self.config.ACTION_REPEAT)

            if self.queue is not None:
                self.queue.put(episode)
            else:
                yield episode

            if self.is_terminated.value:
                break

        if self.queue is not None:
            self.queue.put(None)
        else:
            yield None

    def generate_transition_for_vectorized_env(self):
        observations, infos = self.train_env.reset(return_info=True)

        if self.agent.is_recurrent_model:
            self.agent.model.init_recurrent_hidden()

        while True:
            if self.agent.is_recurrent_model:
                observations = [(observations, self.agent.model.recurrent_hidden)]

            if self.config.ACTION_MASKING:
                unavailable_actions = []
                for env_id in range(self.train_env.num_envs):
                    unavailable_actions.append(infos[env_id]["unavailable_actions"])
            else:
                unavailable_actions = None

            actions, scaled_actions = self._get_action(observations, unavailable_actions)

            next_observations, rewards, dones, infos = self.train_env.step(scaled_actions)

            for env_id, (observation, action, next_observation, reward, done, info) in enumerate(
                    zip(observations, actions, next_observations, rewards, dones, infos)
            ):
                self.total_time_step += 1

                info["actor_id"] = self.actor_id
                info["env_id"] = env_id

                self.histories[env_id].append(Transition(
                    observation=observation,
                    action=action,
                    next_observation=next_observation,
                    reward=reward,
                    done=done,
                    info=info
                ))

                if self.working_actor:
                    self.last_train_env_info = info

                if len(self.histories[env_id]) == self.config.N_STEP or done:
                    n_step_transition, real_n_steps = Actor.get_n_step_transition(
                        history=self.histories[env_id], env_id=env_id,
                        actor_id=self.actor_id, info=info, done=done, total_time_step=self.total_time_step,
                        config=self.config
                    )

                    if self.working_actor:
                        self.episode_rewards[env_id] += n_step_transition.reward
                        self.agent.buffer.append(n_step_transition)

                        self.queue.put({
                            "message_type": "TRANSITION",
                            "done": done,
                            "episode_reward": self.episode_rewards[env_id],
                            "last_train_env_info": self.last_train_env_info,
                            "real_n_steps": real_n_steps
                        })
                        if done:
                            self.episode_rewards[0] = 0.0
                    else:
                        if self.queue is not None:
                            self.queue.put(n_step_transition)
                        else:
                            yield n_step_transition

            observations = next_observations

            if self.working_actor and len(self.agent.buffer) >= self.config.BATCH_SIZE:
                self.working_train()

            if self.is_terminated.value:
                break

        if self.queue is not None:
            self.queue.put(None)
        else:
            yield None

    def _get_action(self, observations, unavailable_actions, vectorized_env=True):
        if self.config.ACTION_MASKING:
            actions = self.agent.get_action(obs=observations, unavailable_actions=unavailable_actions)
        else:
            actions = self.agent.get_action(obs=observations)

        if isinstance(self.agent.action_space, Discrete):
            scaled_actions = actions
        elif isinstance(self.agent.action_space, Box):
            scaled_actions = actions * self.agent.action_scale + self.agent.action_bias
        else:
            raise ValueError()

        if vectorized_env:
            return actions, scaled_actions
        else:
            return actions[0], scaled_actions[0]

    @staticmethod
    def get_n_step_transition(history, env_id, actor_id, info, done, total_time_step, config):
        n_step_transitions = tuple(history)
        next_observation = n_step_transitions[-1].next_observation

        n_step_reward = 0.0
        for n_step_transition in reversed(n_step_transitions):
            n_step_reward = n_step_transition.reward + \
                            config.GAMMA * n_step_reward * (0.0 if n_step_transition.done else 1.0)
            # if n_step_transition.done:
            #     break

        info["actor_id"] = actor_id
        info["env_id"] = env_id
        info["real_n_steps"] = len(n_step_transitions)

        if type(n_step_transitions[0].observation) == LazyFrames:
            observation = np.asarray(n_step_transitions[0].observation)
        else:
            observation = n_step_transitions[0].observation

        if type(n_step_transitions[0].observation) == LazyFrames:
            next_observation = np.asarray(next_observation)

        n_step_transition = Transition(
            observation=observation,
            action=n_step_transitions[0].action,
            next_observation=next_observation,
            reward=n_step_reward,
            done=done,
            info=info
        )

        history.clear()

        return n_step_transition, info["real_n_steps"]

    def working_train(self):
        self.agent._before_train()
        self.agent.shared_model_access_lock.acquire()
        count_training_steps = self.agent.worker_train()
        self.agent.shared_model_access_lock.release()
        self.agent.buffer.clear()                 # ON_POLICY!
        self.agent._after_train()

        self.training_step += count_training_steps
        self.next_train_time_step += self.config.TRAIN_INTERVAL_GLOBAL_TIME_STEPS

        self.queue.put({
            "message_type": "TRAIN",
            "count_training_steps": count_training_steps,
            "last_train_env_info": self.last_train_env_info
        })