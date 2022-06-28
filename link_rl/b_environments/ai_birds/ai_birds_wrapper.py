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
		self.train_mode = train_mode

		self.rl_client = ClientRLAgent(train_mode=self.train_mode)

		self.new_obs_size = 84
		self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(3, self.new_obs_size, self.new_obs_size))
		self.action_space = spaces.Discrete(50)

		self.pass_initial_phase = False

		self.current_game_level = -1

		self.NUMBER_OF_LEVELS = 320

	def get_observation(self, raw_state):
		obs = cv2.resize(src=raw_state, dsize=(self.new_obs_size, self.new_obs_size), interpolation=cv2.INTER_AREA)
		obs = np.transpose(obs, axes=(2, 0, 1))
		return obs

	def proceed_initial_phase(self):
		print("##### INITIAL_PHASE: BEGIN #####")
		self.rl_client.ar.configure(self.rl_client.id)

		self.rl_client.ar.set_game_simulation_speed(100)

		game_state = self.rl_client.ar.get_game_state()		# GameState.NEWTRIAL
		print("1. game_state: {0}".format(game_state))

		self.rl_client.ar.ready_for_new_set()

		self.rl_client.ar.get_game_state()		# GameState.NEWTRAININGSET
		print("2. game_state: {0}".format(game_state))

		self.rl_client.ar.ready_for_new_set()

		main_menu = False
		while not main_menu:
			game_state = self.rl_client.ar.get_game_state()		# GameState.LOADING, GameState.LOADING, ...., GameState.MAIN_MENU
			print("3. game_state: {0}".format(game_state))
			if game_state == GameState.MAIN_MENU:
				main_menu = True

		self.current_game_level = self.rl_client.ar.load_next_available_level()
		self.rl_client.level_count += 1
		self.rl_client.novelty_existence = self.rl_client.ar.get_novelty_info()

		self.pass_initial_phase = True
		print("##### INITIAL_PHASE: END #####")

	def reset(self, *, seed: Optional[int] = None, return_info: bool = False, options: Optional[dict] = None):
		if not self.pass_initial_phase:
			self.proceed_initial_phase()

		game_state = self.rl_client.ar.get_game_state()   # GameState.PLAYING
		self.rl_client.get_slingshot_center()
		raw_state, ground_truth = self.rl_client.ar.get_ground_truth_with_screenshot()
		observation = self.get_observation(raw_state)

		info = self.get_info(game_state)

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

		info = self.get_info(game_state)

		if game_state in [GameState.WON, GameState.LOST]:
			done = True

			self.current_game_level = self.rl_client.ar.load_next_available_level()
			self.rl_client.level_count += 1
		else:
			done = False

		reward = self.get_reward(reward, game_state)

		# [NOTE] game_state 가 GameState.EVALUATION_TERMINATED 가 되기 전에 미리 전체 초기화
		# 이유: GameState.EVALUATION_TERMINATED 일 때에는 WON 인지 LOST인지 파악 불가
		if self.train_mode and self.current_game_level == self.NUMBER_OF_LEVELS:
			self.rl_client = ClientRLAgent(train_mode=self.train_mode)
			self.pass_initial_phase = False

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
			print(game_state, "!!!")
			raise ValueError()

		return reward

	def get_info(self, game_state):
		info = {
			"game_state": game_state,
			"current_game_level": self.current_game_level
		}

		return info

	def render(self, mode="human"):
		pass


class Dummy_Agent:
	def __init__(self):
		self.IS_IN_TRAINING_MODE = True
		self.shoots_before_level_is_completed = 0
		self.TOTAL_ACTIONS = 50  # 0=10, 50=60

	def get_action(self, observation):
		assert observation is not None

		action = np.random.randint(0, 50)
		# convert 0-50 to 10-60
		action += 10
		return action


def run_env():
	env = AIBirdsWrapper()
	agent = Dummy_Agent()

	total_episodes = 10_000
	total_time_steps = 1

	for ep in range(1, total_episodes + 1):
		observation, info = env.reset(return_info=True)
		print("RESET!!!", info)
		done = False
		episode_time_steps = 1

		while not done:
			action = agent.get_action(observation)
			next_observation, reward, done, info = env.step(action)
			print("[Ep.: {0:4}, Time Steps: {1:4}/{2:4}] "
				  "Obs.: {3}, Action: {4}, Next_obs.: {5}, Reward: {6:.4f}, Done: {7}, Info: {8}".format(
				ep, episode_time_steps, total_time_steps,
				observation.shape, action, next_observation.shape, reward, done, info
			))
			observation = next_observation
			total_time_steps += 1
			episode_time_steps += 1


if __name__ == "__main__":
	run_env()
