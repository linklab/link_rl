import gym
import torch
from config.names import EnvironmentName
from common.environments import Environment


class Pendulum_v0(Environment):
    def __init__(self):
        self.env = gym.make(EnvironmentName.PENDULUM_V0.value)
        super(Pendulum_v0, self).__init__()
        self.action_shape = self.get_action_shape()
        self.state_shape = self.get_state_shape()

        self.continuous = True
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

        self.last_episode_reward = None

        self.n_states = self.get_n_states()
        self.n_actions = self.get_n_actions()

    def get_n_states(self):
        n_states = self.env.observation_space.shape[0]
        return n_states

    def get_n_actions(self):
        n_actions = self.env.action_space.shape[0]
        return n_actions

    def get_state_shape(self):
        state_shape = self.env.observation_space.shape
        return state_shape

    def get_action_shape(self):
        action_shape = (self.env.action_space.shape[0],)
        return action_shape

    def get_action_space(self):
        return self.env.action_space

    @property
    def action_meanings(self):
        action_meanings = ["Joint effort",]
        return action_meanings

    def reset(self):
        state = self.env.reset()
        return state

    def step(self, action):
        next_state, reward, done, info = self.env.step(action)
        if type(reward) == torch.Tensor:
            reward = reward.item()
        adjusted_reward = reward

        # return next_state, reward, adjusted_reward, done, info

        return next_state, reward, done, info

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()
