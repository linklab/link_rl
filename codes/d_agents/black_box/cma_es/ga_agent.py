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
        self.elite = None
        self.global_steps = 0

        self.solved = False

    def initialize(self, env):
        self.env = env

        self.population = [
            (chromosome, self.evaluate(chromosome)) for chromosome in self.chromosomes
        ]

        self.population.sort(key=lambda p: p[1], reverse=True)
        self.elite = self.population[0]

    def evaluate(self, model):
        sum_episode_reward = 0.0
        for _ in range(self.params.EVALUATION_RUNS):
            observation = self.env.reset()
            while True:
                observation_v = torch.FloatTensor([observation]).to(self.device)
                action_prob = model(observation_v)
                acts = action_prob.max(dim=1)[1]
                observation, reward, done, _ = self.env.step(acts.data.cpu().numpy()[0])
                sum_episode_reward += reward
                if done:
                    break
        return sum_episode_reward / self.params.EVALUATION_RUNS

    def mutation(self, model):
        new_model = copy.deepcopy(model)
        for parameter in new_model.parameters():
            noise = np.random.normal(size=parameter.data.size())
            noise_v = torch.FloatTensor(noise).to(self.device)
            parameter.data += self.params.NOISE_STANDARD_DEVIATION * noise_v
        return new_model

    def next_generation(self):
        # generate next population
        prev_population = self.population
        self.population = []

        if self.elite:
            fitness = self.evaluate(self.elite[0])
            self.population.append((self.elite[0], fitness))

        for _ in range(self.params.POPULATION_SIZE - 1):
            parent_chromosome_idx = np.random.randint(0, self.params.COUNT_FROM_PARENTS)
            parent_chromosome = prev_population[parent_chromosome_idx][0]
            mutated_chromosome = self.mutation(parent_chromosome)
            fitness = self.evaluate(mutated_chromosome)
            self.population.append((mutated_chromosome, fitness))

        self.population.sort(key=lambda p: p[1], reverse=True)
        self.elite = self.population[0]
        self.set_best_chromosome_to_model()

    def set_best_chromosome_to_model(self):
        self.model.load_state_dict(self.elite[0].state_dict())

    def __call__(self, states, agent_states=None):
        states = states[0]
        action_prob = self.model(states)
        acts = action_prob.max(dim=1)[1]

        return acts.data.cpu().numpy(), None
