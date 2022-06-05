import numpy as np
from dm_control.suite import ALL_TASKS

from dm_control import viewer
import b_environments.wrappers.dm_control as dmc2gym

# print(*ALL_TASKS, sep="\n")
from gym.spaces import Discrete, Box


def print_all_dmc_env_info():
	for id, (domain_name, task_name) in enumerate(ALL_TASKS):
		env = dmc2gym.make(domain_name=domain_name, task_name=task_name, seed=1)
		observation_space = env.observation_space
		action_space = env.action_space

		env_spec = env.spec

		observation_space_str = "OBS_SPACE: {0}, SHAPE: {1}".format(
			type(observation_space), observation_space.shape
		)

		action_space_str = "ACTION_SPACE: {0}, SHAPE: {1}".format(
			type(action_space), action_space.shape
		)

		if isinstance(action_space, Discrete):
			action_space_str += ", N: {0}".format(action_space.n)
		elif isinstance(action_space, Box):
			action_space_str += ", RANGE: {0}".format(action_space)

		print("{0:2}: Domain Name: {1:>12}, Task Name: {2:>14} | reward_threshold: {3} | {4:55} {5}".format(
			id + 1, domain_name, task_name, env_spec.reward_threshold,
			observation_space_str, action_space_str
		))


def print_sample_execution():
	env = dmc2gym.make(domain_name="humanoid", task_name="stand", seed=1)

	# Define a uniform random policy.
	def get_action(obs):
		actions = env.action_space.sample()
		actions_np = np.asarray(actions)
		return actions_np

	# Launch the viewer application.
	viewer.launch(env.original_env, policy=get_action)


if __name__ == "__main__":
	#print_all_dmc_env_info()
	print_sample_execution()

