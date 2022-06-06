import time

import numpy as np
from dm_control.suite import ALL_TASKS

from dm_control import viewer
import b_environments.wrappers.dm_control as dmc_gym

# print(*ALL_TASKS, sep="\n")
from gym.spaces import Discrete, Box


def print_all_dmc_env_info(from_pixels=False):
	for id, (domain_name, task_name) in enumerate(ALL_TASKS):
		if from_pixels:
			env = dmc_gym.make(
				domain_name=domain_name, task_name=task_name, seed=1, from_pixels=True, visualize_reward=False
			)
		else:
			env = dmc_gym.make(domain_name=domain_name, task_name=task_name, seed=2)

		observation_space = env.observation_space
		action_space = env.action_space
		env_spec = env.spec

		observation_space_str = "OBS_SPACE: {0}, SHAPE: {1}".format(type(observation_space), observation_space.shape)
		action_space_str = "ACTION_SPACE: {0}, SHAPE: {1}".format(type(action_space), action_space.shape)

		if isinstance(action_space, Discrete):
			action_space_str += ", N: {0}".format(action_space.n)
		elif isinstance(action_space, Box):
			action_space_str += ", RANGE: {0}".format(action_space)
		else:
			raise ValueError()

		print("{0:2}: Domain Name: {1:>12}, Task Name: {2:>14} | reward_threshold: {3} | {4:55} {5}".format(
			id + 1, domain_name, task_name, env_spec.reward_threshold, observation_space_str, action_space_str
		))
		del env
	print()


def dummy_agent_test(from_pixels=False):
	domain_name = "cartpole"
	task_name = "balance"

	if from_pixels:
		env = dmc_gym.make(
			domain_name=domain_name, task_name=task_name, seed=1, from_pixels=True, visualize_reward=False
		)
	else:
		env = dmc_gym.make(domain_name=domain_name, task_name=task_name, seed=2)

	class Dummy_Agent:
		def get_action(self, observation):
			assert observation is not None
			actions = env.action_space.sample()
			return actions

	agent = Dummy_Agent()

	for i in range(100):
		observation = env.reset()
		done = False

		while not done:
			time.sleep(0.05)
			action = agent.get_action(observation)
			next_observation, reward, done, info = env.step(action)
			# env.render()
			print("Observation: {0}, Action: {1}, next_observation: {2}, Reward: {3}, Done: {4}, Info: {5}".format(
				observation, action, next_observation, reward, done, info
			))
			observation = next_observation

		time.sleep(1)


def play_test():
	# env = dmc_gym.make(domain_name="humanoid", task_name="stand", seed=1)
	env = dmc_gym.make(domain_name="cartpole", task_name="three_poles", seed=1)
	# Define a uniform random policy.
	def get_action(obs):
		actions = env.action_space.sample()
		actions_np = np.asarray(actions)
		return actions_np

	# Launch the viewer application.
	viewer.launch(env.original_env, policy=get_action)


if __name__ == "__main__":
	#print_all_dmc_env_info(from_pixels=True)
	#print_all_dmc_env_info(from_pixels=False)

	dummy_agent_test(from_pixels=True)
	dummy_agent_test(from_pixels=False)

	#play_test()

