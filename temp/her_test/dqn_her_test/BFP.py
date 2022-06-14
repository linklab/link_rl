import torch
from copy import deepcopy


class BFP:
    def __init__(self, N):
        self.N = N
        self.state = None
        self.goal = None

    @staticmethod
    def get_state_and_goal(state, goal):
        return torch.cat((state, goal), dim=1)

    def reset(self):
        self.state = torch.rand((1, self.N)).round()
        self.goal = torch.rand((1, self.N)).round()

        info = {}
        info["desired_goal"] = self.goal
        info["achieved_goal"] = self.state

        state_and_goal = self.get_state_and_goal(self.state, info["desired_goal"])

        return state_and_goal, info

    def step(self, action):
        self.state[0, action] = 1.0 - self.state[0, action]
        reward = -1.0
        done = False

        dist = (self.state - self.goal).abs().sum()
        if dist == 0:
            reward = 0.0
            done = True

        info = {}
        info["desired_goal"] = self.goal
        info["achieved_goal"] = self.state
        info["dist"] = dist

        state_and_goal = self.get_state_and_goal(self.state, info["desired_goal"])

        return deepcopy(state_and_goal), reward, done, info


class BFP_OLD:
    def __init__(self,N):
        self.N = N
        self.state = None
        self.goal = None

    def reset(self):
        self.state = torch.rand((1, self.N)).round()
        self.goal = torch.rand((1, self.N)).round()

        self.state = torch.cat((self.state, self.goal), dim=1)

        info = {}
        info["desired_goal"] = self.goal
        info["achieved_goal"] = self.state

        return self.state, info

    def step(self, action):
        self.state[0, action] = 1.0 - self.state[0, action]
        reward = -1.0
        done = False

        if (self.state[0, 0:self.N] - self.state[0, self.N:]).abs().sum() == 0:
            reward = 0.0
            done = True

        dist = (self.state[0, 0:self.N] - self.state[0, self.N:]).abs().sum()

        return deepcopy(self.state), reward, done, dist
