import gym
import time

from link_rl.b_environments.wrapper import CustomActionWrapper, CustomRewardWrapper, CustomObservationWrapper


class SleepyToyEnv(gym.Env):
    """
    Environment with observation 0..3 and actions 0..2
    """

    def __init__(self):
        super(SleepyToyEnv, self).__init__()
        self.observation_space = gym.spaces.Discrete(n=4)  # 0, 1, 2, 3
        self.action_space = gym.spaces.Discrete(n=3) # 0, 1, 2

        self.current_state = -1
        self.terminal_state = 4

    def reset(self):
        self.current_state = 0
        #print("RESET!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        return self.current_state

    def step(self, action):
        assert self.action_space.contains(action), \
            "Action {0} is not valid".format(action)

        time.sleep(action)

        self.current_state += action

        is_done = self.current_state >= self.terminal_state
        if is_done:
            reward = 10.0
            return None, reward, True, {}
        else:
            reward = 0.0
            return self.current_state, reward, False, {}

    def render(self, mode="human"):
        pass

    def close(self):
        pass


def make_sleepy_toy_env():
    # env = SleepyToyEnv()

    env = CustomActionWrapper(CustomRewardWrapper(CustomObservationWrapper(
        SleepyToyEnv()
    )))
    return env

