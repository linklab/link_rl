import gym
import evogym.envs
from evogym import sample_robot
from gym.spaces import Discrete, Box


def print_info():
	idx = 1
	for env_spec in gym.envs.registry.all():
		if "evogym" in env_spec.entry_point:
			body, connections = sample_robot((5, 5))
			env = gym.make(env_spec.id, body=body)
			observation_space = env.observation_space
			action_space = env.action_space
			env_spec = env.spec

			observation_space_str = "OBS_SPACE: {0}, SHAPE: {1} - ".format(
				type(observation_space), observation_space.shape,
			)
			action_space_str = "ACTION_SPACE: {0}, SHAPE: {1}".format(type(action_space), action_space.shape)

			if isinstance(action_space, Discrete):
				action_space_str += ", N: {0}".format(action_space.n)
			elif isinstance(action_space, Box):
				action_space_str += ", RANGE: {0}".format(action_space)
			else:
				raise ValueError()

			print("{0:2}: env_id: {1:>22} | {2:60} | {3}".format(
				idx, env_spec.id, observation_space_str, action_space_str
			))
			idx += 1


def proceed():
	robot_structure, robot_connections = sample_robot((3, 3))
	env = gym.make('Walker-v0', body=robot_structure, connections=robot_connections)

	print(robot_structure)
	print(robot_connections)
	print(env.observation_space.shape, env.action_space.shape, "!!!! - 1")

	#body, connections = sample_robot((5, 5))
	env = gym.make('Walker-v0', body=robot_structure)

	print(env.observation_space.shape, env.action_space.shape, "!!!! - 2")

	#body, connections = sample_robot((5, 5))
	env = gym.make('Walker-v0', body=robot_structure)

	print(env.observation_space.shape, env.action_space.shape, "!!!! - 3")

	steps(env)


def proceed_2():
	from evogym import EvoWorld, EvoSim, EvoViewer, sample_robot
	world = EvoWorld()

	robot_structure, robot_connections = sample_robot((5, 5))

	world.add_from_array(
		name='robot',
		structure=robot_structure,
		x=3,
		y=1,
		connections=robot_connections
	)

	sim = EvoSim(world)

	print(sim.VOXEL_SIZE)
	print(sim.get_dim_action_space(robot_name="robot"))
	print(sim.get_actuator_indices(robot_name="robot"))

	viewer = EvoViewer(sim)
	viewer.track_objects('robot', 'box')


def steps(env):
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


if __name__ == '__main__':
	#print_info()
	proceed()
	#proceed_2()