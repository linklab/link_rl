from gym import core, spaces
from dm_control import suite
from dm_env import specs
import numpy as np
from collections import deque
import cv2


def _spec_to_box(spec, dtype):
	def extract_min_max(s):
		assert s.dtype == np.float64 or s.dtype == np.float32
		dim = np.int(np.prod(s.shape))
		if type(s) == specs.Array:
			bound = np.inf * np.ones(dim, dtype=np.float32)
			return -bound, bound
		elif type(s) == specs.BoundedArray:
			zeros = np.zeros(dim, dtype=np.float32)
			return s.minimum + zeros, s.maximum + zeros

	mins, maxs = [], []
	for s in spec:
		mn, mx = extract_min_max(s)
		mins.append(mn)
		maxs.append(mx)
	low = np.concatenate(mins, axis=0).astype(dtype)
	high = np.concatenate(maxs, axis=0).astype(dtype)
	assert low.shape == high.shape
	return spaces.Box(low, high, dtype=dtype)


def _flatten_obs(obs):
	obs_pieces = []
	for v in obs.values():
		flat = np.array([v]) if np.isscalar(v) else v.ravel()
		obs_pieces.append(flat)
	return np.concatenate(obs_pieces, axis=0)


class DMCWrapper(core.Env):
	def __init__(
			self,
			domain_name,
			task_name,
			task_kwargs=None,
			visualize_reward={},
			from_pixels=False,
			height=84,
			width=84,
			camera_id=0,
			frame_skip=4,
			environment_kwargs=None,
			frame_stack=1,
			grayscale=True
	):
		super(DMCWrapper, self).__init__()
		#assert 'random' in task_kwargs, 'please specify a seed, for deterministic behaviour'
		self._from_pixels = from_pixels
		self._height = height
		self._width = width
		self._camera_id = camera_id
		self._frame_skip = frame_skip
		self._frame_stack = frame_stack
		self._grayscale = grayscale

		# create task
		self.original_env = suite.load(
			domain_name=domain_name,
			task_name=task_name,
			task_kwargs=task_kwargs,
			visualize_reward=visualize_reward,
			environment_kwargs=environment_kwargs
		)

		# true and normalized action spaces
		self._true_action_space = _spec_to_box([self.original_env.action_spec()], np.float32)
		self._norm_action_space = spaces.Box(
			low=-1.0,
			high=1.0,
			shape=self._true_action_space.shape,
			dtype=np.float32
		)

		# create observation space
		if from_pixels:
			assert isinstance(frame_stack, int) and frame_stack > 0
			if self._grayscale:
				shape = [self._frame_stack, height, width]
			else:
				shape = [3 * self._frame_stack, height, width]
			self._observation_space = spaces.Box(
				low=0, high=1, shape=shape, dtype=np.uint8
			)
			self._frames = deque([], maxlen=self._frame_stack)
		else:
			self._observation_space = _spec_to_box(
				self.original_env.observation_spec().values(),
				np.float64
			)

		self._state_space = _spec_to_box(
			self.original_env.observation_spec().values(),
			np.float64
		)

		# self.current_state = None

		# set seed
		self.seed(seed=task_kwargs.get('seed', 1))

	def __getattr__(self, name):
		return getattr(self.original_env, name)

	def get_observation(self, time_step, first_step=False):
		if self._from_pixels:
			obs = self.render(
				height=self._height,
				width=self._width,
				camera_id=self._camera_id
			)
			if self._grayscale:
				obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
				obs = np.expand_dims(obs, 0)
			else:
				obs = obs.transpose(2, 0, 1).copy()
			##############FRAME_STACK###############
			if first_step:
				for _ in range(self._frame_stack):
					self._frames.append(obs)
				obs = self._transform_observation()
			else:
				self._frames.append(obs)
				obs = self._transform_observation()
			########################################
			obs = obs / 255.0
		else:
			obs = _flatten_obs(time_step.observation)
		return obs

	def _convert_action(self, action):
		action = action.astype(np.float64)
		true_delta = self._true_action_space.high - self._true_action_space.low
		norm_delta = self._norm_action_space.high - self._norm_action_space.low
		action = (action - self._norm_action_space.low) / norm_delta
		action = action * true_delta + self._true_action_space.low
		action = action.astype(np.float32)
		return action

	@property
	def observation_space(self):
		return self._observation_space

	@property
	def state_space(self):
		return self._state_space

	@property
	def action_space(self):
		return self._norm_action_space

	@property
	def reward_range(self):
		return 0, self._frame_skip

	def seed(self, seed):
		self._true_action_space.seed(seed)
		self._norm_action_space.seed(seed)
		self._observation_space.seed(seed)

	def step(self, action):
		#assert self._norm_action_space.contains(action)
		action = self._convert_action(action)
		#assert self._true_action_space.contains(action)
		reward = 0
		extra = {'internal_state': self.original_env.physics.get_state().copy()}

		for _ in range(self._frame_skip):
			time_step = self.original_env.step(action)
			reward += time_step.reward or 0
			done = time_step.last()
			if done:
				break
		obs = self.get_observation(time_step, first_step=False)
		# self.current_state = _flatten_obs(time_step.observation)
		extra['discount'] = time_step.discount
		return obs, reward, done, extra

	def reset(self, return_info=False):
		time_step = self.original_env.reset()
		# self.current_state = _flatten_obs(time_step.observation)
		obs = self.get_observation(time_step, first_step=True)
		
		if return_info:
			return obs, None
		else:
			return obs

	def render(self, mode='rgb_array', height=None, width=None, camera_id=0):
		assert mode == 'rgb_array', 'only support rgb_array mode, given %s' % mode
		height = height or self._height
		width = width or self._width
		camera_id = camera_id or self._camera_id
		return self.original_env.physics.render(
			height=height, width=width, camera_id=camera_id
		)

	def close(self):
		self.original_env.close()

	def _transform_observation(self):
		assert len(self._frames) == self._frame_stack
		obs = np.concatenate(list(self._frames), axis=0)
		return obs
