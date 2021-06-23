"""basic wrappers, useful for reinforcement learning on gym envs"""
# Mostly copy-pasted from https://github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py
import time

import numpy as np
from collections import deque
import gym
from gym import spaces
from skimage.transform import resize
from skimage.color import rgb2gray


class NoopResetEnv(gym.Wrapper):
    def __init__(self, env=None, noop_max=30):
        """Sample initial states by taking random number of no-ops on reset.
        No-op is assumed to be action 0.
        """
        super(NoopResetEnv, self).__init__(env)
        self.noop_max = noop_max
        self.override_num_noops = None
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def step(self, action):
        return self.env.step(action)

    def reset(self):
        """ Do no-op action for a number of steps in [1, noop_max]."""
        self.env.reset()
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = np.random.randint(1, self.noop_max + 1)
        assert noops > 0
        obs = None
        for _ in range(noops):
            obs, _, done, _ = self.env.step(0)
            if done:
                obs = self.env.reset()
        return obs


class FireResetEnv(gym.Wrapper):
    def __init__(self, env=None):
        """For environments where the user need to press FIRE for the game to start."""
        super(FireResetEnv, self).__init__(env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def step(self, action):
        return self.env.step(action)

    def reset(self):
        self.env.reset()
        obs, _, done, _ = self.env.step(action=1)
        if done:
            self.env.reset()
        # obs, _, done, _ = self.env.step(action=2)
        # if done:
        #     self.env.reset()
        return obs


class EpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env=None):
        """Make end-of-life == end-of-episode, but only reset on true game over.
        Done by DeepMind for the DQN and co. since it helps value estimation.
        """
        super(EpisodicLifeEnv, self).__init__(env)
        self.lives = 0
        self.was_real_done = True
        self.was_real_reset = False

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = done
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives and lives > 0:
            # for Qbert somtimes we stay in lives == 0 condtion for a few frames
            # so its important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            done = True
        self.lives = lives
        return obs, reward, done, info

    def reset(self):
        """Reset only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        """
        if self.was_real_done:
            obs = self.env.reset()
            self.was_real_reset = True
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, _ = self.env.step(0)
            self.was_real_reset = False
        self.lives = self.env.unwrapped.ale.lives()
        return obs


class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env=None, skip=4):
        """Return only every `skip`-th frame"""
        super(MaxAndSkipEnv, self).__init__(env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = deque(maxlen=2)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = None
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            self._obs_buffer.append(obs)
            total_reward += reward
            if done:
                break

        max_frame = np.max(np.stack(self._obs_buffer), axis=0)

        return max_frame, total_reward, done, info

    def reset(self):
        """Clear past frame buffer and init. to first obs. from inner env."""
        self._obs_buffer.clear()
        obs = self.env.reset()
        self._obs_buffer.append(obs)
        return obs


class SkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        """Return only every `skip`-th frame"""
        gym.Wrapper.__init__(self, env)
        self._skip       = skip

    def step(self, action):
        """Repeat action, sum reward, and max over last observations."""
        total_reward = 0.0
        done = None
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class ProcessFrame84(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(ProcessFrame84, self).__init__(env)
        self.observation_space = spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)

    def observation(self, obs):
        return ProcessFrame84.process(obs)

    #@staticmethod
    # def process(frame):
    #     if frame.size == 210 * 160 * 3:
    #         img = np.reshape(frame, [210, 160, 3]).astype(np.float32)
    #     elif frame.size == 250 * 160 * 3:
    #         img = np.reshape(frame, [250, 160, 3]).astype(np.float32)
    #     else:
    #         assert False, "Unknown resolution."
    #     img = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114
    #     resized_screen = cv2.resize(img, (84, 110), interpolation=cv2.INTER_AREA)
    #     x_t = resized_screen[18:102, :]
    #     x_t = np.reshape(x_t, [84, 84, 1])
    #     return x_t.astype(np.uint8)

    @staticmethod
    def process(frame):
        '''입력데이터 전처리.

        Args:
            frame(np.array): 받아온 이미지를 그레이 스케일링 후 84X84로 크기변경
                그리고 정수값으로 저장하기위해(메모리 효율 높이기 위해) 255를 곱함

        Returns:
            np.array: 변경된 이미지
        '''
        # 바로 전 frame과 비교하여 max를 취함으로써 flickering을 제거
        # x = np.maximum(X, X1)
        # 그레이 스케일링과 리사이징을 하여 데이터 크기 수정
        x = np.uint8(resize(rgb2gray(frame), (84, 84, 1), mode='reflect') * 255)
        return x


class ClippedRewardsWrapper(gym.RewardWrapper):
    def reward(self, reward):
        """Change all the positive rewards to 1, negative to -1 and keep zero."""
        return np.sign(reward)

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        return ob, self.reward(reward), done, info


class LazyFrames(object):
    def __init__(self, frames):
        """This object ensures that common frames between the observations are only stored once.
        It exists purely to optimize memory usage which can be huge for DQN's 1M frames replay
        buffers.
        This object should only be converted to numpy array before being passed to the model.
        You'd not believe how complex the previous solution was."""
        self._frames = frames
        self._out = None

    def _force(self):
        if self._out is None:
            self._out = np.concatenate(self._frames, axis=0)
            self._frames = None
        return self._out

    def __array__(self, dtype=None):
        out = self._force()
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        return len(self._force())

    def __getitem__(self, i):
        return self._force()[i]

    def count(self):
        frames = self._force()
        return frames.shape[1:frames.ndim]

    def frame(self, i):
        return self._force()[i, ...]


class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        """Stack k last frames.
        Returns lazy array, which is much more memory efficient.
        See Also
        --------
        baselines.common.atari_wrappers.LazyFrames
        """
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = spaces.Box(low=0, high=255, shape=(shp[0]*k, shp[1], shp[2]), dtype=np.uint8)

    def reset(self):
        ob = self.env.reset()
        for _ in range(self.k):
            self.frames.append(ob)
        return self._get_ob()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.frames.append(ob)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        assert len(self.frames) == self.k
        return LazyFrames(list(self.frames))


class ScaledFloatFrame(gym.ObservationWrapper):
    def observation(self, obs):
        # careful! This undoes the memory optimization, use
        # with smaller replay buffers only.
        return np.array(obs).astype(np.float32) / 255.0


class ImageToPyTorch(gym.ObservationWrapper):
    """
    Change image shape to CWH
    """
    def __init__(self, env):
        super(ImageToPyTorch, self).__init__(env)
        old_shape = self.observation_space.shape
        # self.observation_space = gym.spaces.Box(
        #     low=0.0, high=1.0, shape=(old_shape[-1], old_shape[0], old_shape[1]), dtype=np.float32
        # )

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(old_shape[-1], old_shape[0], old_shape[1]), dtype=np.uint8
        )

    def observation(self, observation):
        return np.swapaxes(observation, 2, 0)


def process_experience(exp, buffer):
    assert np.array_equal(exp.state.__array__()[1, :, :], exp.last_state.__array__()[0, :, :])
    assert np.array_equal(exp.state.__array__()[2, :, :], exp.last_state.__array__()[1, :, :])
    assert np.array_equal(exp.state.__array__()[3, :, :], exp.last_state.__array__()[2, :, :])

    history = np.zeros([5, 84, 84], dtype=np.uint8)
    history[0, :, :] = exp.state.__array__()[0, :, :]
    for i in range(1, 4):
        history[i, :, :] = exp.state.__array__()[i, :, :]

    if exp.last_state is not None:
        history[4, :, :] = exp.last_state.__array__()[3, :, :]

    buffer.add_sample((history, exp.action, exp.reward, exp.last_state is None))


def wrap_dqn(env, stack_frames=4, episodic_life=True, reward_clipping=True):
    """Apply a common set of wrappers for Atari games."""
    assert 'NoFrameskip' in env.spec.id
    if episodic_life:
        env = EpisodicLifeEnv(env)
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    # env = SkipEnv(env, skip=4)
    if 'FIRE' in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = ProcessFrame84(env)
    # env = ScaledFloatFrame(env)
    env = ImageToPyTorch(env)
    env = FrameStack(env, k=stack_frames)
    if reward_clipping:
        env = ClippedRewardsWrapper(env)
    return env


def wrap_super_mario_bros(env, stack_frames=3, reward_clipping=True):
    env = ProcessFrame84(env)
    env = ImageToPyTorch(env)
    env = FrameStack(env, k=stack_frames)
    return env