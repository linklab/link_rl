import time

import gym
from gym.spaces import Discrete, Box
from gym import envs


def print_all_gym_robotics_env_info():
	for idx, env_spec in enumerate(envs.registry.all()):
		if idx > 63:
			break
		env = gym.make(env_spec.id)
		observation_space = env.observation_space
		action_space = env.action_space
		env_spec = env.spec

		observation_space_str = "OBS_SPACE: {0}, SHAPE: {1} [ACHIEVED_GOAL_SPACE: {2}, SHAPE: {3}] [DESIRED_GOAL_SPACE: {4}, SHAPE: {5}]".format(
			type(observation_space["observation"]), observation_space["observation"].shape,
			type(observation_space["achieved_goal"]), observation_space["achieved_goal"].shape,
			type(observation_space["desired_goal"]), observation_space["desired_goal"].shape,
		)
		action_space_str = "ACTION_SPACE: {0}, SHAPE: {1}".format(type(action_space), action_space.shape)

		if isinstance(action_space, Discrete):
			action_space_str += ", N: {0}".format(action_space.n)
		elif isinstance(action_space, Box):
			action_space_str += ", RANGE: {0}".format(action_space)
		else:
			raise ValueError()

		print("{0:2}: env_id: {1:>50} | {2:60} | {3}".format(
			idx + 1, env_spec.id, observation_space_str, action_space_str
		))
		del env
	print()


def dummy_agent_test(render=False):
	env_id = "HandManipulateBlockRotateXYZ-v0"
	env = gym.make(env_id)

	class Dummy_Agent:
		def get_action(self, observation):
			assert observation is not None
			actions = env.action_space.sample()
			return actions

	agent = Dummy_Agent()

	for i in range(100):
		observation = env.reset()
		done = False
		env.render()
		while not done:
			time.sleep(0.05)
			action = agent.get_action(observation)
			next_observation, reward, done, info = env.step(action)
			# env.render()
			print("Observation: {0}, Action: {1}, next_observation: {2}, Reward: {3}, Done: {4}, Info: {5}".format(
				observation, action, next_observation, reward, done, info
			))
			observation = next_observation
			env.render()

		time.sleep(1)


if __name__ == "__main__":
	#print_all_gym_robotics_env_info()
	dummy_agent_test(render=True)


