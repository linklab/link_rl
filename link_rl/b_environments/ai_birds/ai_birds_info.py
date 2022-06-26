import math
import numpy as np
import time

from link_rl.b_environments.ai_birds.utils.agent_client import GameState
from link_rl.b_environments.ai_birds.utils.client_rl_agent import ClientRLAgent, StateMaker
from link_rl.b_environments.ai_birds.utils.point2D import Point2D

TOTAL_STEPS = 1000
OFFSET = 0
INPUT_HEIGHT = 84
INPUT_WIDTH = 84


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

		a = np.random.randint(0, 49)
		# convert 0-50 to 10-60
		a += 10
		# Convert simulator coordinates to pixels...

		return a


class LinkBird:
	def __init__(self):
		# TRAINING/ EVALUATION SWITCH Parameters
		self.IS_IN_TRAINING_MODE = True  	# indicates if the agent is in training mode, switching it off will stop agent from training

		self.TRAIN_LEVELS_NUMBER = 1501  	# total number of train levels
		self.TRAINING_SCORES = np.zeros([self.TRAIN_LEVELS_NUMBER, 3])  # 300 training levels, for each level we have: level score, Won or Not?, Num of birds used

		self.EVAL_LEVELS_NUMBER = 51  		# total number of eval levels
		self.EVAL_SCORES = np.zeros([self.EVAL_LEVELS_NUMBER, 3])  # 50 evaluation levels, for each level we have: level score, Won or Not?, Num of birds used

		# TRAINING_SET_TIMES = 0 # num of times all train levels were played
		# EVAL_SET_TIMES = -1 # num of times all eval levels were played

		# self.agent = DQNAgent(input_shape=InputShape(INPUT_HEIGHT, INPUT_WIDTH), n_actions=50, batch_size=BATCH_SIZE)

		self.agent = Dummy_Agent()

		self.rl_client = ClientRLAgent()

		self.state_maker = StateMaker()

		self.shoots_before_level_is_completed = 0

	def print_game_state(self, game_state, env_step):
		print("▶▶▶ Game State: {0:30} | IS_IN_TRAINING_MODE: {1} | rl_client.level_count: {2} | env_step: {3} ◀◀◀".format(
			game_state, self.IS_IN_TRAINING_MODE, self.rl_client.level_count, env_step
		))

	def run(self, run_id):
		highest_total_score_TRAIN = 0
		highest_total_score_EVAL = 0

		try:
			info = self.rl_client.agent_client.configure(self.rl_client.id)
			self.rl_client.agent_client.set_game_simulation_speed(100)

			self.rl_client.solved = [0 for x in range(self.rl_client.agent_client.get_number_of_levels())]
			# print("self.rl_client.solved:", self.rl_client.solved)

			scores = self.rl_client.agent_client.get_all_level_scores()
			max_scores = np.zeros([len(self.rl_client.solved)])

			self.rl_client.agent_client.load_next_available_level()
			# self.rl_client.level_count += 1

			s = None
			s_previous = None
			r_previous = 0
			a_previous = 0
			r_average = 0
			r_total = 0
			r = 0
			all_levels_played_count = 0
			d = False

			first_time_in_level_in_episode = True

			for env_step in range(1, TOTAL_STEPS):
				game_state = self.rl_client.agent_client.get_game_state()

				self.print_game_state(game_state=game_state, env_step=env_step)

				# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■ #
				if game_state == GameState.REQUESTNOVELTYLIKELIHOOD:
					# Require report novelty likelihood and then playing can be resumed
					# dummy likelihoods:
					novelty_likelihood = 0.9
					non_novelty_likelihood = 0.1
					novel_obj_ids = {1, -2, -398879789}
					novelty_level = 0
					novelty_description = ""
					self.rl_client.agent_client.report_novelty_likelihood(
						novelty_likelihood, non_novelty_likelihood, novel_obj_ids, novelty_level, novelty_description
					)

				elif game_state in [
					GameState.NEWTRIAL, GameState.NEWTESTSET, GameState.NEWTRAININGSET, GameState.RESUMETRAINING
				]:
					self.IS_IN_TRAINING_MODE = True
					s = None
					s_previous = None
					r_previous = 0
					a_previous = 0
					r_average = 0
					r_total = 0
					r = 0
					all_levels_played_count = 0
					d = False

					first_time_in_level_in_episode = True
					if game_state in [GameState.NEWTESTSET]:
						self.IS_IN_TRAINING_MODE = False
					elif game_state in [GameState.NEWTRAININGSET, GameState.RESUMETRAINING]:
						self.IS_IN_TRAINING_MODE = True
					(time_limit, interaction_limit, n_levels, attempts_per_level, mode, seq_or_set, allowNoveltyInfo) \
						= self.rl_client.agent_client.ready_for_new_set()

					print("time_limit: {0}, interaction_limit: {1}, n_levels: {2}, attempts_per_level: {3}, "
						  "mode: {4}, seq_or_set: {5}, allowNoveltyInfo: {6}".format(
						time_limit, interaction_limit, n_levels, attempts_per_level, mode, seq_or_set, allowNoveltyInfo
					))

				elif game_state == GameState.EVALUATION_TERMINATED:
					# store info and disconnect the agent as the evaluation is finished
					print("Done evaluating")
					exit(1)
				# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■ #

				# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■ #
				if self.IS_IN_TRAINING_MODE is True \
						and game_state in [GameState.NEWTRAININGSET, GameState.RESUMETRAINING]:
					# Training
					# EVAL_SET_TIMES += 1
					if highest_total_score_EVAL < self.EVAL_SCORES[:, 0].sum(0):
						highest_total_score_EVAL = self.EVAL_SCORES[:, 0].sum(0)

					print("[Training mode] interactions so far: {0}, current total score of evaluation set: {1}, "
						  "highest ever total score: {2}, run id: {3}".format(
						env_step,
						self.EVAL_SCORES[:, 0].sum(0),
						highest_total_score_EVAL,
						run_id
					))

					s = None
					s_previous = None
					r_previous = 0
					a_previous = 0
					r_total = 0
					self.EVAL_SCORES = np.zeros([self.EVAL_LEVELS_NUMBER, 3])

				# rl_client.agent_client.restart_level()
				# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■ #

				# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■ #
				if self.IS_IN_TRAINING_MODE is False and game_state == GameState.NEWTESTSET:
					# EVALUATION
					# TRAINING_SET_TIMES += 1
					if highest_total_score_TRAIN < self.TRAINING_SCORES[:, 0].sum(0):
						highest_total_score_TRAIN = self.TRAINING_SCORES[:, 0].sum(0)

					print("[Evaluating Agent] interactions so far: {0}, current total score of training set: {1}, "
						  "highest ever total score: {2}, run id: {3}".format(
						env_step,
						self.TRAINING_SCORES[:, 0].sum(0),
						highest_total_score_TRAIN,
						run_id
					))

					s = None
					s_previous = None
					r_previous = 0
					a_previous = 0
					r_total = 0
					self.TRAINING_SCORES = np.zeros([self.TRAIN_LEVELS_NUMBER, 3])
				# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■ #

				# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■ #
				if s is not None and game_state in [GameState.PLAYING, GameState.WON, GameState.LOST]:
					# save previous state
					s_previous = s
					r_previous = r
					a_previous = a
				# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■ #

				# Done (win or lose)
				# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■ #
				if game_state in [GameState.WON, GameState.LOST]:
					self.shoots_before_level_is_completed = 0
					# save current state before reloading the level
					# r = rl_client.agent_client.get_current_score()
					s = self.rl_client.agent_client.do_screenshot()
					s = self.state_maker.make(s)

					# check for change of number of levels in the game
					self.rl_client.update_no_of_levels()
					# scores = self.rl_client.agent_client.get_all_level_scores()
					self.rl_client.check_my_score()

					if game_state == GameState.WON:
						is_win = 1

						self.rl_client.agent_client.load_next_available_level()
						self.rl_client.level_count += 1

						r_previous *= 1
						r_total += r_previous
					else:  # GameState.LOST
						is_win = 0

						# If lost, then restart the level
						self.rl_client.failed_counter += 1
						if self.rl_client.failed_counter >= 0:  # for testing , go directly to the next level
							self.rl_client.failed_counter = 0
							self.rl_client.agent_client.load_next_available_level()
							self.rl_client.level_count += 1
						else:
							# print("restart")
							self.rl_client.agent_client.restart_level()

						r_previous *= -1

					if self.IS_IN_TRAINING_MODE == True:
						self.TRAINING_SCORES[self.rl_client.level_count, :] = \
							[self.rl_client.agent_client.get_current_score(), is_win, OFFSET + env_step]
					else:
						self.EVAL_SCORES[self.rl_client.level_count, :] = \
							[self.rl_client.agent_client.get_current_score(), is_win, OFFSET + env_step]

					d = 1
					first_time_in_level_in_episode = True
				# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■ #

				# Action 추론 및 실행
				# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■ #
				if game_state == GameState.PLAYING:
					# Start of the episode
					# r = rl_client.agent_client.get_current_score()

					# rl_client.get_slingshot_center()
					# game_state = rl_client.agent_client.get_game_state()
					#
					# s, img = rl_client.agent_client.get_ground_truth_with_screenshot()

					if first_time_in_level_in_episode:
						# If first time in level reset states so we dont
						# carry previous states with us
						s = None
						s_previous = None
						self.rl_client.agent_client.fully_zoom_out()  # Needed as we depend on fully zoom out values
						self.rl_client.agent_client.fully_zoom_out()
						self.rl_client.agent_client.fully_zoom_out()
						self.rl_client.agent_client.fully_zoom_out()
						self.rl_client.agent_client.fully_zoom_out()
						self.rl_client.agent_client.fully_zoom_out()
						self.rl_client.agent_client.fully_zoom_out()
						first_time_in_level_in_episode = False

					self.rl_client.get_slingshot_center()

					raw_state, ground_truth = self.rl_client.agent_client.get_ground_truth_with_screenshot()

					# print("[Ground Truth] Type: {0}, Contents: {1}\n".format(type(ground_truth), ground_truth))
					# t1 = rl_client.agent_client.get_ground_truth_without_screenshot()
					# t1, t2 = rl_client.agent_client.get_noisy_ground_truth_with_screenshot()
					# t1 = rl_client.agent_client.get_noisy_ground_truth_without_screenshot()
					# t1 = rl_client.agent_client.do_screenshot()

					s = self.state_maker.make(raw_state)
					# print("[Raw State] Type: {0}, Shape: {1}, [Transformed State] Type: {0}, Shape: {1}".format(
					# 	type(raw_state), raw_state.shape, type(s), s.shape
					# ))
					a = self.agent.get_action(s)

					# convert 0-50 to 10-60
					# a += 10
					self.step(a)
				# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■ #

				elif game_state == GameState.LEVEL_SELECTION:
					print("unexpected level selection page, go to the last current level")
					self.shoots_before_level_is_completed = 0
					self.rl_client.agent_client.load_next_available_level()
					self.rl_client.level_count += 1
					self.rl_client.novelty_existence = self.rl_client.agent_client.get_novelty_info()

				elif game_state == GameState.MAIN_MENU:
					print("unexpected main menu page, reload the level : ", self.rl_client.level_count)
					self.shoots_before_level_is_completed = 0
					self.rl_client.agent_client.load_next_available_level()
					self.rl_client.level_count += 1
					self.rl_client.novelty_existence = self.rl_client.agent_client.get_novelty_info()
				# time.sleep(10)

				elif game_state == GameState.EPISODE_MENU:
					print("unexpected main menu page, reload the level : ", self.rl_client.level_count)
					self.shoots_before_level_is_completed = 0
					self.rl_client.agent_client.load_next_available_level()
					self.rl_client.level_count += 1
					self.rl_client.novelty_existence = self.rl_client.agent_client.get_novelty_info()

				# 버퍼에 트랜지션 저장 및 일정 주기마다 훈련
				# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■ #
				if self.IS_IN_TRAINING_MODE is True and \
						game_state in [GameState.PLAYING, GameState.WON, GameState.LOST]:
					pass
				# if (d and game_state == GameState.WON):
				#     r_previous = 1
				# elif (d and game_state == GameState.LOST):
				#     r_previous = 0
				# else:
				#     r_previous = 0

				# r_previous = r_previous / 150000

				# 	# 버퍼에 트랜지션 저장
				# if s_previous is not None:
				# 	self.agent.memory.push(
				# 		s_previous,
				# 		a_previous,
				# 		s,
				# 		torch.tensor([r_previous], device=device),
				# 		d
				# 	)

				# # 일정 주기마다 훈련
				# if env_step % UPDATE_FREQUENCY == 0 and len(self.agent.memory.memory) > BATCH_SIZE:
				# 	self.agent.optimize()
				#
				# 	if self.rl_client.level_count % 50 == 0:
				# 		print("Total score over levels: " + str(r_total) + " run id: " + str(run_id))
				# 		r_total = 0
				# 		all_levels_played_count += 1
				#
				# 		self.agent.save_model()
				#
				# if env_step % TARGET_UPDATE_FREQUENCY == 0:
				# 	self.agent.target_net.load_state_dict(self.agent.policy_net.state_dict())
		# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■ #

		except Exception as e:
			print("Error: ", e)
		finally:
			time.sleep(10)

	def reset(self):
		pass

	def step(self, int_action):
		# Convert simulator coordinates to pixels...
		release_point = self.rl_client.tp.find_release_point(self.rl_client.sling_mbr, math.radians(int_action))
		tap_time = int(1250)

		# Execute a in the environment
		if self.shoots_before_level_is_completed > 20:
			# 20 shots for a level... likely novelty 3 flip x, skip level
			print("20 shots for a level... likely novelty 3 flip x, skip level")
			self.rl_client.agent_client.load_next_available_level()
			self.rl_client.level_count += 1

		if not release_point:
			# Add logic to deal with unreachable target
			print("No release point is found")
			release_point = Point2D(
				-int(40 * math.cos(math.radians(int_action))), int(40 * math.sin(math.radians(int_action)))
			)

		dx = int(release_point.X - self.rl_client.sling_center.X)
		dy = int(release_point.Y - self.rl_client.sling_center.Y)
		# print("[Sling Center] X: {0}, Y: {1}".format(
		# 	self.rl_client.sling_center.X, self.rl_client.sling_center.Y
		# ))
		# print("[Release Point] X: {0}, Y: {1} \t [Tap Time] {2}".format(
		# 	release_point.X, release_point.Y, tap_time
		# ))
		self.rl_client.agent_client.shoot_and_record_ground_truth(release_point.X, release_point.Y, 0, tap_time, 1)
		reward = self.rl_client.agent_client.get_current_score()

		self.shoots_before_level_is_completed += 1


if __name__ == "__main__":
	link_bird = LinkBird()
	link_bird.run(run_id=0)
