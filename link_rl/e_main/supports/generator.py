from collections import deque

import numpy as np
from gym.spaces import Discrete, Box

from link_rl.e_main.supports.actor import Actor
from link_rl.g_utils.commons_rl import Episode
from link_rl.g_utils.types import Transition


class TransitionGenerator:
	def __init__(self, train_env, agent, is_recurrent_model, is_terminated, config):
		self.train_env = train_env
		self.agent = agent
		self.is_recurrent_model = is_recurrent_model
		self.is_terminated = is_terminated
		self.config = config

		self.histories = []
		for _ in range(self.config.N_VECTORIZED_ENVS):
			self.histories.append(deque(maxlen=self.config.N_STEP))

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

	def generate_transition_for_single_env(self):
		observation, info = self.train_env.reset(return_info=True)

		actor_time_step = 1

		while True:
			observations = np.expand_dims(observation, axis=0)

			if self.is_recurrent_model:
				self.agent.model.init_recurrent_hidden()
				observations = [(observations, self.agent.model.recurrent_hidden)]

			if self.config.ACTION_MASKING:
				unavailable_actions = []
				unavailable_actions.append(info["unavailable_actions"])
			else:
				unavailable_actions = None

			action, scaled_action = self._get_action(observations, unavailable_actions, vectorized_env=False)

			next_observation, reward, done, info = self.train_env.step(scaled_action)

			info["actor_id"] = 0
			info["env_id"] = 0
			info["actor_time_step"] = actor_time_step

			self.histories[0].append(Transition(
				observation=observation,
				action=action,
				next_observation=next_observation,
				reward=reward,
				done=done,
				info=info
			))
			if len(self.histories[0]) == self.config.N_STEP or done:
				n_step_transition = Actor.get_n_step_transition(
					history=self.histories[0], env_id=0,
					actor_id=0, info=info, done=done, config=self.config
				)
				yield n_step_transition

			if done:
				next_observation, info = self.train_env.reset(return_info=True)

			observation = next_observation

			if self.is_terminated.value:
				break
			else:
				actor_time_step += 1

		yield None

	def generate_episode_for_single_env(self):
		actor_time_step = 0
		step = 0
		while True:
			actor_time_step += 1

			# Collect trajectory
			obs = self.train_env.reset()
			episode = Episode(self.config, obs, self.agent.n_out_actions)
			while not episode.done:
				action = self.agent.get_action(obs, step=step, t0=episode.first)
				obs, reward, done, info = self.train_env.step(action.cpu().numpy())
				info["actor_id"] = 0
				info["env_id"] = 0
				info["actor_time_step"] = actor_time_step
				episode += (obs, action, reward, done, info)
			assert len(episode) == int(1000 / self.config.ACTION_REPEAT)
			step += int(1000 / self.config.ACTION_REPEAT)
			yield episode

	def generate_transition_for_vectorized_env(self):
		observations, infos = self.train_env.reset(return_info=True)

		actor_time_step = 1

		while True:
			if self.is_recurrent_model:
				self.agent.model.init_recurrent_hidden()
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
			else:
				actor_time_step += 1

		yield None
