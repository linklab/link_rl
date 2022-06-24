import gym
import numpy as np
import random

GAME_NAME_LIST = [
    'running-competition', 'wrestling', 'football', 'table-hockey'
]


class DummyCompetitionOlympicsAgent:
	def __init__(self):
		self.force_range = [-100, 200]
		self.angle_range = [-30, 30]

	def seed(self, seed=None):
		random.seed(seed)

	def get_action(self, obs):
		force = random.uniform(self.force_range[0], self.force_range[1])
		angle = random.uniform(self.angle_range[0], self.angle_range[1])

		return [[force], [angle]]


class CompetitionOlympicsEnvWrapper(gym.Wrapper):
	metadata = {}

	def __init__(self, env, env_render=False, controlled_agent_index=1, agent=None):
		super().__init__(env)

		self.env_render = env_render
		self.controlled_agent_index = controlled_agent_index
		self.observation_space = gym.spaces.Box(
			low=-np.inf, high=np.inf, shape=(2, 40, 40), dtype=np.float32
		)
		self.action_space = gym.spaces.Box(
			low=np.asarray([-100., -30.0]), high=np.asarray([200., 30.0]), shape=(2,), dtype=float
		)

		# self.action_space = gym.spaces.Box(
		# 	low=np.asarray([[-100.], [-30.0]]), high=np.asarray([[200.], [30.0]]), shape=(2, 1), dtype=float
		# )

		self.opponent_agent = DummyCompetitionOlympicsAgent()
		self.last_observation_opponent_agent = None

	def _get_normalize_observation(self, observation, game_name):
		if game_name == 'running-competition':
			observation = np.concatenate((observation, np.full(shape=(1, 40, 40), fill_value=0.0)), axis=0)
		elif game_name == 'wrestling':
			observation = np.concatenate((observation, np.full(shape=(1, 40, 40), fill_value=2.5)), axis=0)
		elif game_name == 'football':
			observation = np.concatenate((observation, np.full(shape=(1, 40, 40), fill_value=5.0)), axis=0)
		elif game_name == 'table-hockey':
			observation = np.concatenate((observation, np.full(shape=(1, 40, 40), fill_value=7.5)), axis=0)
		else:
			raise ValueError()

		return observation / 10.0

	def reset(self, return_info=False):
		observation = self.env.reset()
		observation_opponent_agent = np.array(observation[1 - self.controlled_agent_index]['obs']['agent_obs'])
		observation_controlled_agent = np.expand_dims(
			observation[self.controlled_agent_index]['obs']['agent_obs'], axis=0
		)
		observation_controlled_agent = self._get_normalize_observation(
			observation_controlled_agent, self.env.env_core.current_game.game_name
		)

		self.last_observation_opponent_agent = observation_opponent_agent

		if return_info:
			info = {"game_name": self.env.env_core.current_game.game_name}
			return observation_controlled_agent, info
		else:
			return observation_controlled_agent

	def step(self, action_controlled):
		if self.env_render:
			self.render()
		action_controlled = np.expand_dims(action_controlled, axis=1)
		action_opponent = self.opponent_agent.get_action(self.last_observation_opponent_agent)
		action = [action_opponent, action_controlled] if self.controlled_agent_index == 1 else [action_controlled, action_opponent]

		next_observation, reward, done, info_before, info_after = self.env.step(action)

		next_observation_opponent_agent = next_observation[1 - self.controlled_agent_index]['obs']['agent_obs']
		next_observation_controlled_agent = np.expand_dims(
			next_observation[self.controlled_agent_index]['obs']['agent_obs'], axis=0
		)

		self.last_observation_opponent_agent = next_observation_opponent_agent

		next_observation_controlled_agent = self._get_normalize_observation(
			next_observation_controlled_agent, self.env.env_core.current_game.game_name
		)

		reward_controlled = self._get_reward_shaped(reward, done, self.controlled_agent_index)
		#reward_opponent = reward[1 - self.controlled_agent_index]

		info = {
			"game_name": self.env.env_core.current_game.game_name,
			"original_reward": reward
		}
		if done:
			if reward[self.controlled_agent_index] > reward[1 - self.controlled_agent_index]:
				info['win_controlled_agent'] = True
			elif reward[self.controlled_agent_index] < reward[1 - self.controlled_agent_index]:
				info['win_opponent_agent'] = True
			elif reward[self.controlled_agent_index] == reward[1 - self.controlled_agent_index]:
				info['draw'] = True
			else:
				raise ValueError()

		return next_observation_controlled_agent, reward_controlled, done, info

	def _get_reward_shaped(self, reward, done, controlled_agent_index):
		if done and reward[0] != reward[1]:
			reward_shaped = [0.0, 10.0] if reward[0] < reward[1] else [10.0, 0.0]
			#print("-" * 10, reward, reward_shaped, reward_shaped[controlled_agent_index])
		else:
			# if reward[0] != reward[1]:
			# 	print("-" * 10, reward, reward[controlled_agent_index])
			reward_shaped = reward

		return reward_shaped[controlled_agent_index]

	# def _get_reward_shaped(self, reward, done, controlled_agent_index):
	# 	if not done:
	# 		reward_shaped = [-1., -1.]
	# 	else:
	# 		if reward[0] != reward[1]:
	# 			reward_shaped = [reward[0] - 100, reward[1]] if reward[0] < reward[1] else [reward[0], reward[1] - 100]
	# 		else:
	# 			reward_shaped = [-1., -1.]
	# 	return reward_shaped[controlled_agent_index]

	def render(self, mode='human'):
		self.env.env_core.render()

	def close(self):
		pass
