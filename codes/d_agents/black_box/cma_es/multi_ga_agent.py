import numpy as np
import copy

import torch
from codes.c_models.discrete_action.simple_model import SimpleModel
from codes.d_agents.a0_base_agent import BaseAgent


# Deep Neuroevolution: Genetic Algorithms are a Competitive
# Alternative for Training Deep Neural Networks for Reinforcement Learning
class AgentMultiGA(BaseAgent):
    def __init__(self, worker_id, input_shape, num_outputs, params, device):
        super(AgentMultiGA, self).__init__(worker_id, params, device)
        self.__name__ = "AgentMultiGA"

        self.input_shape = input_shape
        self.num_outputs = num_outputs

        self.model = SimpleModel(
            worker_id=worker_id,
            input_shape=input_shape,
            num_outputs=num_outputs,
            params=params,
            device=device
        ).to(self.device)

    def initialize(self, env):
        self.env = env

    def evaluate(self, model):
        observation = self.env.reset()
        episode_reward = 0.0
        steps = 0
        while True:
            observation_v = torch.FloatTensor([observation]).to(self.device)
            action_prob = model(observation_v)
            acts = action_prob.max(dim=1)[1]
            observation, reward, done, _ = self.env.step(acts.data.cpu().numpy()[0])
            episode_reward += reward
            steps += 1
            if done:
                break
        return episode_reward, steps

    def mutate(self, chromosome, seed, copy_chromosome=True):
        new_chromosome = copy.deepcopy(chromosome) if copy_chromosome else chromosome
        np.random.seed(seed)
        for parameter in new_chromosome.parameters():
            noise = np.random.normal(size=parameter.data.size())
            noise_v = torch.FloatTensor(noise)
            parameter.data += self.params.NOISE_STANDARD_DEVIATION * noise_v
        return new_chromosome

    def build(self, seeds):
        torch.manual_seed(seeds[0])
        chromosome = SimpleModel(
            worker_id=self.worker_id,
            input_shape=self.input_shape,
            num_outputs=self.num_outputs,
            params=self.params,
            device=self.device
        ).to(self.device)

        for seed in seeds[1:]:
            chromosome = self.mutate(chromosome, seed, copy_chromosome=False)

        return chromosome

    def __call__(self, states, agent_states=None):
        states = states[0]
        action_prob = self.model(states)
        acts = action_prob.max(dim=1)[1]

        return acts.data.cpu().numpy(), None
