### java 실행 (별도의 터미널 두 개에서 각각 동시 실행)
#### Train 환경
# java -jar game_playing_interface.jar --game-start-port 9001 --agent-port 2004
#### Test 환경
# java -jar game_playing_interface.jar --game-start-port 9011 --agent-port 2014

import math
from typing import Optional, Union, Tuple

import cv2
import numpy as np
import gym
from gym import spaces

from link_rl.b_environments.ai_birds.src.ver_0_5_13.client.agent_client import GameState
from link_rl.b_environments.ai_birds.src.ver_0_5_13.demo.ddqn_test_harness_agent import ClientRLAgent


class AIBirdsWrapper(gym.Env):
	def __init__(self, train_mode=True):
		super(AIBirdsWrapper, self).__init__()
		self.rl_client = ClientRLAgent(train_mode=train_mode)

		self.new_obs_size = 84
		self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(3, self.new_obs_size, self.new_obs_size))
		self.action_space = spaces.Discrete(50)

		self.pass_initial_phase = False

	def get_observation(self, raw_state):
		obs = cv2.resize(src=raw_state, dsize=(self.new_obs_size, self.new_obs_size), interpolation=cv2.INTER_AREA)
		obs = np.transpose(obs, axes=(2, 0, 1))
		return obs

	def proceed_initial_phase(self):
		self.rl_client.ar.configure(self.rl_client.id)

		self.rl_client.ar.set_game_simulation_speed(100)
		self.rl_client.ar.load_next_available_level()

		self.rl_client.ar.get_game_state()		# GameState.NEWTRIAL
		self.rl_client.ar.ready_for_new_set()

		self.rl_client.ar.get_game_state()		# GameState.NEWTRAININGSET
		self.rl_client.ar.ready_for_new_set()

		main_menu = False
		while not main_menu:
			game_state = self.rl_client.ar.get_game_state()		# GameState.LOADING, GameState.LOADING, ...., GameState.MAIN_MENU
			if game_state == GameState.MAIN_MENU:
				main_menu = True

		self.rl_client.ar.load_next_available_level()
		self.rl_client.level_count += 1
		self.rl_client.novelty_existence = self.rl_client.ar.get_novelty_info()

		self.pass_initial_phase = True

	def reset(self, *, seed: Optional[int] = None, return_info: bool = False, options: Optional[dict] = None):
		if not self.pass_initial_phase:
			self.proceed_initial_phase()

		game_state = self.rl_client.ar.get_game_state()   # GameState.PLAYING
		self.rl_client.get_slingshot_center()
		raw_state, ground_truth = self.rl_client.ar.get_ground_truth_with_screenshot()
		observation = self.get_observation(raw_state)

		info = {
			"game_state": game_state,
			"level_count": self.rl_client.level_count
		}

		if return_info:
			return observation, info
		else:
			return observation

	def step(self, action):
		release_point = self.rl_client.tp.find_release_point(self.rl_client.sling_mbr, math.radians(action))
		tap_time = int(1250)

		self.rl_client.ar.shoot_and_record_ground_truth(release_point.X, release_point.Y, 0, tap_time, 1)

		raw_state, ground_truth = self.rl_client.ar.get_ground_truth_with_screenshot()
		observation = self.get_observation(raw_state)
		reward = self.rl_client.ar.get_current_score()

		game_state = self.rl_client.ar.get_game_state()   # GameState.PLAYING

		info = {
			"game_state": game_state,
			"level_count": self.rl_client.level_count
		}

		reward = self.get_reward(reward, game_state)

		if game_state in [GameState.WON, GameState.LOST]:
			done = True

			# check for change of number of levels in the game
			self.rl_client.update_no_of_levels()
			# scores = self.rl_client.ar.get_all_level_scores()
			self.rl_client.check_my_score()

			self.rl_client.ar.load_next_available_level()
			self.rl_client.level_count += 1
		else:
			done = False

		return observation, reward, done, info

	def get_reward(self, reward, game_state):
		reward = reward / 100_000

		if game_state == GameState.WON:
			reward += 1.0
		elif game_state == GameState.LOST:
			pass
		elif game_state == GameState.PLAYING:
			pass
		else:
			raise ValueError()

		return reward

	def render(self, mode="human"):
		pass


class Dummy_Agent:
	def __init__(self):
		self.IS_IN_TRAINING_MODE = True
		self.shoots_before_level_is_completed = 0
		self.TOTAL_ACTIONS = 50  # 0=10, 50=60

	def reset_agent(self):
		self.IS_IN_TRAINING_MODE = True  # indicates if the agent is in training mode, switching it off will stop agent from training
		pass

	def get_action(self, observation):
		assert observation is not None

		action = np.random.randint(0, 50)
		# convert 0-50 to 10-60
		action += 10
		return action


def run_env():
	env = AIBirdsWrapper()
	agent = Dummy_Agent()

	for i in range(100):
		observation, info = env.reset(return_info=True)
		print("RESET!!!", info)
		done = False

		while not done:
			action = agent.get_action(observation)
			next_observation, reward, done, info = env.step(action)
			print("Observation: {0}, Action: {1}, next_observation: {2}, Reward: {3}, Done: {4}, Info: {5}".format(
				observation.shape, action, next_observation.shape, reward, done, info
			))
			observation = next_observation


if __name__ == "__main__":
	run_env()
