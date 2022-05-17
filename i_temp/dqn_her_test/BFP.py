import torch
from copy import deepcopy


class BFP:
    def __init__(self,N):
        self.N = N

    def reset(self):
        state = torch.rand((1, self.N)).round()
        goal = torch.rand((1, self.N)).round()
        done = False
        return torch.cat((state, goal), dim=1), done

    def step(self, x, action):
        y = deepcopy(x)
        y[0, action] = 1.0 - y[0, action]
        reward = -1.0
        done = False
        if (y[0, 0:self.N] - y[0, self.N:]).abs().sum() == 0:
            reward = 0.0
            done = True
        dist = (y[0, 0:self.N] - y[0, self.N:]).abs().sum()
        return y, reward, done, dist