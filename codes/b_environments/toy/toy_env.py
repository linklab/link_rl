import gym


class ToyEnv(gym.Env):
    """
    Environment with observation 0..4 and actions 0..2
    Observations are rotated sequentialy mod 5, reward is equal to given action.
    Episodes are having fixed length of 10
    """

    def __init__(self):
        super(ToyEnv, self).__init__()
        self.observation_space = gym.spaces.Discrete(n=10)
        self.action_space = gym.spaces.Discrete(n=2)
        self.current_state = -1
        self.terminal_state = 9

    def reset(self):
        self.current_state = 0
        return self.current_state

    def step(self, action):
        self.current_state += 1
        is_done = self.current_state == self.terminal_state
        if is_done:
            reward = 10.0
            return self.terminal_state, reward, True, {}
        else:
            reward = -1
            return self.current_state, reward, False, {}
