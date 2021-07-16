import random

import numpy as np
import copy

import torch
from codes.c_models.discrete_action.simple_model import SimpleModel
from codes.d_agents.a0_base_agent import BaseAgent


# Deep Neuroevolution: Genetic Algorithms are a Competitive
# Alternative for Training Deep Neural Networks for Reinforcement Learning
class AgentGA(BaseAgent):
    def __init__(self, worker_id, observation_shape, action_shape, num_outputs, action_min, action_max, params, device):
        super(AgentGA, self).__init__(worker_id, params, action_shape, action_min, action_max, device)
        self.__name__ = "AgentGA"

        self.model = SimpleModel(
            worker_id=worker_id,
            observation_shape=observation_shape,
            num_outputs=num_outputs,
            params=params,
            device=device
        ).to(self.device)

        self.chromosomes = [
            SimpleModel(
                worker_id=worker_id,
                observation_shape=observation_shape,
                num_outputs=num_outputs,
                params=params,
                device=device
            ).to(self.device) for _ in range(self.params.POPULATION_SIZE)
        ]

        self.env = None
        self.population = None
        self.elite = None
        self.global_steps = 0

        self.solved = False

    def initialize(self, env):
        self.env = env

        self.population = [
            (chromosome, self.evaluate(chromosome)[0]) for chromosome in self.chromosomes
        ]

    def evaluate(self, model):
        sum_episode_reward = 0.0
        steps = 0
        for _ in range(self.params.EVALUATION_EPISODES):
            observation = self.env.reset()
            while True:
                observation_v = torch.FloatTensor([observation]).to(self.device)
                action_prob = model(observation_v)
                acts = action_prob.max(dim=1)[1]
                observation, reward, done, _ = self.env.step(acts.data.cpu().numpy()[0])
                sum_episode_reward += reward
                steps += 1
                if done:
                    break
        return sum_episode_reward / self.params.EVALUATION_EPISODES, steps

    def selection(self, tsize=10):
        # https://en.wikipedia.org/wiki/Tournament_selection
        # generate next population based on 'tournament selection'
        prev_population = self.population
        self.population = []

        for _ in range(self.params.POPULATION_SIZE):
            candidates = random.sample(prev_population, tsize)
            self.population.append(max(candidates, key=lambda p: p[1]))

    # def selection(self):
    #     # generate next population
    #     prev_population = self.population
    #     self.population = []
    #
    #     if self.elite:
    #         fitness, _ = self.evaluate(self.elite[0])
    #         self.population.append((self.elite[0], fitness))
    #
    #     for _ in range(self.params.POPULATION_SIZE - 1):
    #         parent_chromosome_idx = np.random.randint(0, self.params.COUNT_FROM_PARENTS)
    #         self.population.append(prev_population[parent_chromosome_idx])

    def mutation(self):
        for idx, (chromosome, _) in enumerate(self.population):
            new_chromosome = copy.deepcopy(chromosome)
            for parameter in new_chromosome.parameters():
                noise = np.random.normal(size=parameter.data.size())
                noise_v = torch.FloatTensor(noise).to(self.device)
                parameter.data += self.params.NOISE_STANDARD_DEVIATION * noise_v

            self.population[idx] = (new_chromosome, self.evaluate(new_chromosome)[0])

    def sort_population_and_set_elite(self):
        self.population.sort(key=lambda p: p[1], reverse=True)
        self.elite = self.population[0]
        self.model.load_state_dict(self.elite[0].state_dict())

    def __call__(self, state, agent_state=None):
        state = state[0]
        action_prob = self.model(state)
        acts = action_prob.max(dim=1)[1]

        return acts.data.cpu().numpy(), None
