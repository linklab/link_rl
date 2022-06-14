import gym


class GymRoboticsEnvWrapper(gym.Wrapper):
	def __init__(self, env):
		super().__init__(env)

		self.observation_space = env.observation_space["observation"]
		self.achieved_goal_space = env.observation_space["achieved_goal"]
		self.desired_goal = env.observation_space["desired_goal"]

	def reset(self, return_info=False):
		observation_with_goals = self.env.reset()
		observation = observation_with_goals['observation']

		if return_info:
			info = {}
			info["achieved_goal"] = observation_with_goals['achieved_goal']
			info["desired_goal"] = observation_with_goals['desired_goal']

			return observation, info
		else:
			return observation

	def step(self, action):
		observation_with_goals, reward, done, info = self.env.step(action)

		observation = observation_with_goals['observation']
		info["achieved_goal"] = observation_with_goals['achieved_goal']
		info["desired_goal"] = observation_with_goals['desired_goal']

		return observation, reward, done, info

	def close(self):
		self.env.close()
