import numpy as np
import copy

import torch
from codes.c_models.discrete_action.simple_model import SimpleModel
from codes.d_agents.a0_base_agent import BaseAgent


# Deep Neuroevolution: Genetic Algorithms are a Competitive
# Alternative for Training Deep Neural Networks for Reinforcement Learning
class AgentGA(BaseAgent):
    def __init__(self, worker_id, input_shape, num_outputs, params, device):
        super(AgentGA, self).__init__(worker_id, params, device)
        self.__name__ = "AgentGA"

        self.model = SimpleModel(
            worker_id=worker_id,
            input_shape=input_shape,
            num_outputs=num_outputs,
            params=params,
            device=device
        ).to(self.device)

        self.chromosomes = [
            SimpleModel(
                worker_id=worker_id,
                input_shape=input_shape,
                num_outputs=num_outputs,
                params=params,
                device=device
            ).to(self.device) for _ in range(self.params.POPULATION_SIZE)
        ]

        self.env = None
        self.population = None

    def initialize(self, env):
        self.env = env

        self.population = [
            (chromosome, self.evaluate(chromosome)) for chromosome in self.chromosomes
        ]

    def evaluate(self, model):
        observation = self.env.reset()
        episode_reward = 0.0
        while True:
            observation_v = torch.FloatTensor([observation]).to(self.device)
            action_prob = model(observation_v)
            acts = action_prob.max(dim=1)[1]
            observation, reward, done, _ = self.env.step(acts.data.cpu().numpy()[0])
            episode_reward += reward
            if done:
                break
        return episode_reward

    def mutate_parent_model(self, model):
        new_model = copy.deepcopy(model)
        for parameter in new_model.parameters():
            noise = np.random.normal(size=parameter.data.size())
            noise_v = torch.FloatTensor(noise).to(self.device)
            parameter.data += self.params.NOISE_STANDARD_DEVIATION * noise_v
        return new_model

    def next_generation(self):
        # generate next population
        prev_population = self.population
        self.population = [self.population[0]]
        for _ in range(self.params.POPULATION_SIZE - 1):
            parent_chromosome_idx = np.random.randint(0, self.params.PARENTS_COUNT)
            parent_chromosome = prev_population[parent_chromosome_idx][0]
            mutated_chromosome = self.mutate_parent_model(parent_chromosome)
            fitness = self.evaluate(mutated_chromosome)
            self.population.append((mutated_chromosome, fitness))

    def __call__(self, states, agent_states=None):
        states = states[0]
        action_prob = self.model(states)
        acts = action_prob.max(dim=1)[1]

        return acts.data.cpu().numpy(), None
