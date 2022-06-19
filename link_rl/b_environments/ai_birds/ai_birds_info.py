import random
import math

from link_rl.b_environments.ai_birds.utils.agent_client import GameState
from link_rl.b_environments.ai_birds.utils.client_rl_agent import ClientRLAgent
from link_rl.b_environments.ai_birds.utils.point2D import Point2D


class Dummy_Agent:
	def __init__(self):
		self.IS_IN_TRAINING_MODE = True
		self.shoots_before_level_is_completed = 0

	def reset_agent(self):
		self.IS_IN_TRAINING_MODE = True  # indicates if the agent is in training mode, switching it off will stop agent from training
		pass

	def get_action(self, observation):
		assert observation is not None
		actions = 0
		return actions


def dummy_agent_test():
	rl_client = ClientRLAgent()
	rl_client.agent_client.configure(rl_client.id)
	rl_client.agent_client.set_game_simulation_speed(100)

	rl_client.solved = [0 for x in range(rl_client.agent_client.get_number_of_levels())]
	print("rl_client.solved:", rl_client.solved)

	rl_client.agent_client.load_next_available_level()

	rl_client.level_count += 1
	print("rl_client.level_count:", rl_client.level_count)
	agent = Dummy_Agent()

	TOTAL_STEPS = 1000

	first_time_in_level_in_episode = True

	for env_step in range(1, TOTAL_STEPS):
		game_state = rl_client.agent_client.get_game_state()

		# r = rl_client.agent_client.get_current_score()
		# print("REWARD: " + str(r))

		if game_state == GameState.REQUESTNOVELTYLIKELIHOOD:
			# Require report novelty likelihood and then playing can be resumed
			# dummy likelihoods:
			print("- REQUESTNOVELTYLIKELIHOOD")
			novelty_likelihood = 0.9
			non_novelty_likelihood = 0.1
			novel_obj_ids = {1, -2, -398879789}
			novelty_level = 0
			novelty_description = ""
			rl_client.ar.report_novelty_likelihood(
				novelty_likelihood, non_novelty_likelihood, novel_obj_ids, novelty_level, novelty_description
			)

		elif game_state == GameState.LEVEL_SELECTION:
			print("- LEVEL_SELECTION: unexpected level selection page, go to the last current level")
			agent.shoots_before_level_is_completed = 0
			rl_client.agent_client.load_next_available_level()
			rl_client.level_count += 1
			rl_client.novelty_existence = rl_client.agent_client.get_novelty_info()

		elif game_state == GameState.MAIN_MENU:
			print("- MAIN_MENU: unexpected main menu page, reload the level : ", rl_client.level_count)
			agent.shoots_before_level_is_completed = 0
			rl_client.agent_client.load_next_available_level()
			rl_client.level_count += 1
			rl_client.novelty_existence = rl_client.agent_client.get_novelty_info()

		elif game_state == GameState.EPISODE_MENU:
			print("- EPISODE_MENU: unexpected main menu page, reload the level : ", rl_client.level_count)
			agent.shoots_before_level_is_completed = 0
			rl_client.agent_client.load_next_available_level()
			rl_client.level_count += 1
			rl_client.novelty_existence = rl_client.agent_client.get_novelty_info()

		elif game_state == GameState.NEWTRIAL:
			print("- NEWTRIAL")
			# Make a fresh agent to continue with a new trial (evaluation)
			agent.reset_agent()
			s = 'None'
			s_previous = 'None'
			r_previous = 0
			a_previous = 0
			r_average = 0
			r_total = 0
			r = 0
			all_levels_played_count = 0
			d = False

			first_time_in_level_in_episode = True
			(time_limit, interaction_limit, n_levels, attempts_per_level, mode, seq_or_set, allowNoveltyInfo) \
				= rl_client.agent_client.ready_for_new_set()
		elif game_state == GameState.NEWTESTSET:
			# DO something to clone a test only agent that does not learn
			print("- NEWTESTSET")
			agent.reset_agent()
			s = 'None'
			s_previous = 'None'
			r_previous = 0
			a_previous = 0
			r_average = 0
			r_total = 0
			r = 0
			all_levels_played_count = 0
			d = False

			first_time_in_level_in_episode = True
			agent.IS_IN_TRAINING_MODE = False
			(time_limit, interaction_limit, n_levels, attempts_per_level, mode, seq_or_set,
			 allowNoveltyInfo) = rl_client.ar.ready_for_new_set()
		# rl_client.ar.ready_for_new_set()
		elif game_state == GameState.NEWTRAININGSET:
			# DO something to resume the training agent
			print("- NEWTRAININGSET")
			agent.reset_agent()
			s = 'None'
			s_previous = 'None'
			r_previous = 0
			a_previous = 0
			r_average = 0
			r_total = 0
			r = 0
			all_levels_played_count = 0
			d = False

			first_time_in_level_in_episode = True
			agent.IS_IN_TRAINING_MODE = True
			(time_limit, interaction_limit, n_levels, attempts_per_level, mode, seq_or_set, allowNoveltyInfo) \
				= rl_client.agent_client.ready_for_new_set()
		# rl_client.ar.ready_for_new_set()
		elif game_state == GameState.RESUMETRAINING:
			print("- RESUMETRAINING")
			agent.reset_agent()
			s = 'None'
			s_previous = 'None'
			r_previous = 0
			a_previous = 0
			r_average = 0
			r_total = 0
			r = 0
			all_levels_played_count = 0
			d = False

			first_time_in_level_in_episode = True
			agent.IS_IN_TRAINING_MODE = True
			(time_limit, interaction_limit, n_levels, attempts_per_level, mode, seq_or_set, allowNoveltyInfo) \
				= rl_client.agent_client.ready_for_new_set()
		elif game_state == GameState.EVALUATION_TERMINATED:
			print("- EVALUATION_TERMINATED")
			# store info and disconnect the agent as the evaluation is finished
			print("Done evaluating")
			exit(1)

		elif game_state == GameState.PLAYING:
			# Start of the episode
			print("- PLAYING")
			# r = rl_client.ar.get_current_score()

			# rl_client.get_slingshot_center()
			# game_state = rl_client.ar.get_game_state()
			#
			# s, img = rl_client.ar.get_ground_truth_with_screenshot()

			if first_time_in_level_in_episode:
				# If first time in level reset states so we dont
				# carry previous states with us
				s = 'None'
				s_previous = 'None'
				rl_client.agent_client.fully_zoom_out()  # Needed as we depend on fully zoom out values
				rl_client.agent_client.fully_zoom_out()
				rl_client.agent_client.fully_zoom_out()
				rl_client.agent_client.fully_zoom_out()
				rl_client.agent_client.fully_zoom_out()
				rl_client.agent_client.fully_zoom_out()
				rl_client.agent_client.fully_zoom_out()
				first_time_in_level_in_episode = False

			rl_client.get_slingshot_center()

			img, _ = rl_client.agent_client.get_ground_truth_with_screenshot()

			print(img.shape, "!!!!")

			a = random.randint(0, 49)
			# convert 0-50 to 10-60
			a += 10
			# Convert simulator coordinates to pixels...
			release_point = rl_client.tp.find_release_point(rl_client.sling_mbr, math.radians(a))
			tap_time = int(1250)

			dx = int(release_point.X - rl_client.sling_center.X)
			dy = int(release_point.Y - rl_client.sling_center.Y)
			print("Shoot: " + str(int(rl_client.sling_center.X)) + ", " + str(int(rl_client.sling_center.Y)) + ", " + str(tap_time))
			rl_client.agent_client.shoot_and_record_ground_truth(release_point.X, release_point.Y, 0, tap_time, 1)

		else:
			print("- OTHER STATE: ", game_state)


if __name__ == "__main__":
	dummy_agent_test()

