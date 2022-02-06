import gym


class MCTSRootWrapper(gym.Wrapper):
    def __init__(self, env):
        super(MCTSRootWrapper, self).__init__(env)

    def step(self, action, agent):
        observation, reward, done, info = self.env.step(action)
        info['root'] = agent.root

        return observation, reward, done, info
