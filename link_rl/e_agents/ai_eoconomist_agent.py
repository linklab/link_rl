import torch
import gym.spaces as spaces
from gym.spaces import Discrete, Box
from torch.distributions import Categorical, Normal
import torch.multiprocessing as mp
import numpy as np

from link_rl.h_utils.buffers.buffer import Buffer
from link_rl.e_agents.agent import Agent
from link_rl.h_utils.types import AgentMode


class AgentAiEconomist(Agent):
    def __init__(self, observation_space, action_space, config, need_train=None):
        super(AgentAiEconomist, self).__init__(self._flatten_obs(observation_space["0"]), action_space["0"], config)
        self.observation_space_ag = self._flatten_obs(observation_space["0"])
        self.observation_space_pl = self._flatten_obs(observation_space["p"])
        self.action_space_ag = action_space["0"]
        self.action_space_pl = action_space["p"]

        # buffer two
        if need_train:
            self.buffer = Buffer(observation_space=self.observation_space_ag, action_space=self.action_space_ag, config=self.config)
        else:
            self.buffer = None

        #models
        self.model_ag = self._model_creator.create_model()
        self.model_pl = self._model_creator.create_model()

        # Access
        self.model = self.model_ag

        self.model_ag.eval()
        self.model_pl.eval()

        self.multi_action_mode_ag = False
        self.multi_action_mode_pl = True

        self.agent_ag = {0: AgentA2c(self.model_ag), 1: AgentA2c(self.model_ag)}
        self.agent_pl = AgentA2c(self.model_pl)

    def _flatten_obs(self, obs):
        return spaces.flatten_space(obs)

    def sample_random_action(self, multi_action_mode, action_space, mask):
        """Sample random UNMASKED action(s) for agent."""
        # Return a list of actions: 1 for each action subspace
        if multi_action_mode:
            split_masks = np.split(mask, []) #########
            return [np.random.choice(np.arange(len(m_)), p=m_ / m_.sum()) for m_ in split_masks]

        # Return a single action
        else:
            return np.random.choice(
                np.arange(action_space.n), p=mask / mask.sum()
            )  # possible action 1, unpossible action 0

    def get_action(self, obs, mode=AgentMode.TRAIN):
        """Samples random UNMASKED actions for each agent in obs."""
        actions = {}

        obs = obs[0]

        for a_idx, a_obs in obs.items():
            if a_idx != 'p':
                actions[a_idx] = self.sample_random_action(
                    self.multi_action_mode_ag, self.action_space_ag, a_obs['action_mask']
                )
            else:
                actions[a_idx] = self.sample_random_action(
                    self.multi_action_mode_pl, self.action_space_pl, a_obs['action_mask']
                )

        #print(actions)
        return actions

    def train(self, training_steps_v):
        count_training_steps = 0
        count_training_steps += 1
        return count_training_steps