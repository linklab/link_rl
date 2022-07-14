import gym
import evogym.envs
from evogym import sample_robot

if __name__ == '__main__':
	body, connections = sample_robot((5, 5))
	env = gym.make('Walker-v0', body=body)

	print("!!!!!")
	for ep in range(1, 10):
		observation = env.reset()
		done = False
		step = 1

		while not done:
			action = env.action_space.sample() - 1
			next_observation, reward, done, info = env.step(action)
			print(
				"[EP: {0}, STEP: {1}] Observation: {2}, Action: {3}, next_observation: {4}, Reward: {5:.5f}, Done: {6}".format(
					ep, step, observation.shape, action.shape, next_observation.shape, reward, done
			))
			env.render()
			observation = next_observation
			step += 1

	env.close()
