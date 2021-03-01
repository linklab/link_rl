import numpy as np

import torch
from codes.c_models.discrete_action.simple_model import SimpleModel
from codes.d_agents.a0_base_agent import BaseAgent


# Covariance Matrix Adaptation Evolution Strategy (CMA-ES)
class AgentEMAES(BaseAgent):
    def __init__(self, worker_id, input_shape, num_outputs, params, device):
        super(AgentEMAES, self).__init__(worker_id, params, device)
        self.__name__ = "AgentEMAES"

        self.model = SimpleModel(
            worker_id=worker_id,
            input_shape=input_shape,
            num_outputs=num_outputs,
            params=params,
            device=device
        ).to(device)

    def sample_noise(self):
        noises = []
        neg_noises = []
        for parameter in self.model.parameters():
            noise = np.random.normal(size=parameter.data.size())
            noise_v = torch.FloatTensor(noise)
            noises.append(noise_v)
            neg_noises.append(-noise_v)
        return noises, neg_noises

    def train_step(self, batch_noise, batch_episode_reward):
        batch_episode_reward = np.array(batch_episode_reward)
        normalized_batch_episode_reward = batch_episode_reward - np.mean(batch_episode_reward)
        normalized_std = np.std(normalized_batch_episode_reward)
        if abs(normalized_std) > 1e-6:
            normalized_batch_episode_reward /= normalized_std

        weighted_noises = None
        for noise, normalized_episode_reward in zip(batch_noise, normalized_batch_episode_reward):
            if weighted_noises is None:
                weighted_noises = [normalized_episode_reward * parameter_noise for parameter_noise in noise]
            else:
                for weighted_noise, parameter_noise in zip(weighted_noises, noise):
                    weighted_noise += normalized_episode_reward * parameter_noise

        for parameter, weighted_noise in zip(self.model.parameters(), weighted_noises):
            update = weighted_noise / (len(batch_episode_reward) * self.params.NOISE_STANDARD_DEVIATION)
            parameter.data += self.params.LEARNING_RATE * update
