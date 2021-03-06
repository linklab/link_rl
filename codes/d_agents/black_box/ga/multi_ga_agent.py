import collections
import numpy as np
import copy
import torch
import torch.multiprocessing as mp

from codes.c_models.discrete_action.simple_model import SimpleModel
from codes.d_agents.a0_base_agent import BaseAgent
from codes.e_utils import rl_utils
from codes.e_utils.common_utils import Cache

MessageFromWorker = collections.namedtuple(
    'MessageFromWorker', field_names=['seeds', 'episode_reward', 'steps']
)

MessageFromMaster = collections.namedtuple(
    'MessageFromMaster', field_names=['seeds_lst']
)

# Deep Neuroevolution: Genetic Algorithms are a Competitive
# Alternative for Training Deep Neural Networks for Reinforcement Learning


class AgentMultiGA(BaseAgent):
    def __init__(self, worker_id, observation_shape, action_shape, num_outputs, action_min, action_max, params, device):
        super(AgentMultiGA, self).__init__(worker_id, params, action_shape, action_min, action_max, device)
        self.__name__ = "AgentMultiGA"

        self.model = SimpleModel(
            worker_id=worker_id,
            observation_shape=observation_shape,
            num_outputs=num_outputs,
            params=params,
            device=device
        ).to(self.device)

        self.env = None
        self.population = None
        self.elite = None
        self.ga_operator = None
        self.global_steps = 0
        self.observation_shape = observation_shape
        self.num_outputs = num_outputs

        mp.set_start_method('spawn')

        self.master_to_worker_queue_lst = []
        self.worker_to_master_queue = mp.Queue(maxsize=params.WORKERS_COUNT)
        self.solved = False

    def initialize(self, env):
        self.env = env
        self.ga_operator = GAOperator(
            env=self.env,
            observation_shape=self.observation_shape, num_outputs=self.num_outputs, params=self.params, device=self.device
        )

        for ga_worker_idx in range(self.params.WORKERS_COUNT):
            master_to_worker_queue = mp.Queue(maxsize=1)
            self.master_to_worker_queue_lst.append(master_to_worker_queue)
            worker = mp.Process(
                target=self.worker_func,
                args=(ga_worker_idx, master_to_worker_queue, self.worker_to_master_queue, self.params, self.device)
            )
            worker.start()

            # ?????? seeds ??? seeds_lst ??????
            seeds_lst = []
            # params.NUM_SEEDS_PER_WORKER: 300
            for _ in range(self.params.NUM_SEEDS_PER_WORKER):
                seeds = (np.random.randint(self.params.MAX_SEED),)
                seeds_lst.append(seeds)

            master_to_worker_queue.put(MessageFromMaster(seeds_lst=seeds_lst))

        self._get_population()

    def _get_population(self):
        self.population = []

        # if self.elite:
        #     fitness, _ = self.ga_operator.evaluate(self.elite[1])
        #     self.population.append((self.elite[0], fitness))

        for _ in range(self.params.POPULATION_SIZE):
            message = self.worker_to_master_queue.get()

            self.population.append((message.seeds, message.episode_reward))
            self.global_steps += message.steps

    def sort_population_and_set_elite(self):
        self.population.sort(key=lambda p: p[1], reverse=True)
        best_seeds = self.population[0][0]
        best_fitness = self.population[0][1]

        if self.elite is None or self.elite[0] != best_seeds:
            best_chromosome = self.ga_operator.build(best_seeds)
            self.elite = (best_seeds, best_chromosome, best_fitness)
            self.model.load_state_dict(best_chromosome.state_dict())

    def selection(self):
        # https://en.wikipedia.org/wiki/Fitness_proportionate_selection: Roulette wheel selection
        # generate next population based on 'fitness proportionate selection'
        total = sum(p[1] for p in self.population)
        selection_probs = [p[1]/total for p in self.population]
        new_population_idx = np.random.choice(len(self.population), size=self.params.POPULATION_SIZE, p=selection_probs)

        prev_population = self.population
        self.population = []
        for idx in new_population_idx:
            self.population.append(prev_population[idx])

    def mutation(self):
        # ??? worker????????? ????????? seeds_lst ??????
        for master_to_worker_queue in self.master_to_worker_queue_lst:
            if self.solved:
                master_to_worker_queue.put("Solved")
            else:
                seeds_lst = []
                for _ in range(self.params.NUM_SEEDS_PER_WORKER):
                    parent = np.random.randint(self.params.POPULATION_SIZE)

                    # ????????? ?????? ??????
                    next_seed = np.random.randint(self.params.MAX_SEED)

                    # ??? ????????? parent chromosome??? ?????? ??? ????????? seeds??? ????????? seed??? ????????? ????????? seeds ??????
                    # (1099612850, 3655502209) --> (1099612850, 3655502209, 1087985398)
                    seeds = list(self.population[parent][0]) + [next_seed]
                    seeds_lst.append(tuple(seeds))

                master_to_worker_queue.put(MessageFromMaster(seeds_lst=seeds_lst))

        self._get_population()

    def __call__(self, state, agent_state=None):
        state = state[0]
        action_prob = self.model(state)
        acts = action_prob.max(dim=1)[1]

        return acts.data.cpu().numpy(), None

    @staticmethod
    def worker_func(ga_worker_id, master_to_worker_queue, worker_to_master_queue, params, device):
        env = rl_utils.get_single_environment(params=params)
        observation_shape, action_shape, num_outputs = rl_utils.get_environment_input_output_info(env)

        ga_operator = GAOperator(
            env=env, observation_shape=observation_shape, num_outputs=num_outputs, params=params, device=device
        )

        # Pool of models (chromosomes): minimize the amount of time spent recreating the parameters from the same seeds.
        # Master??? ?????? ???????????? seeds??? ?????? ????????? ??? chromosome ??????
        chromosome_pool = Cache(maxlen=50)

        while True:
            message = master_to_worker_queue.get()

            if type(message) is str and message == "Solved":
                break

            seeds_lst = message.seeds_lst
            # seeds_lst?????? params.NUM_SEEDS_PER_WORKER (300)????????? seeds ??????
            # seeds??? ????????? seed??? ????????? tuple
            assert len(seeds_lst) == params.NUM_SEEDS_PER_WORKER

            idx_lst = []
            for idx, seeds in enumerate(seeds_lst):
                if len(seeds) == 1:
                    # seeds??? 1?????? seed??? ?????? ?????? --> ?????? chromosome ??????
                    chromosome = ga_operator.build(seeds)

                elif len(seeds) > 1:
                    # seeds??? 2??? ????????? seed??? ?????? ??????
                    # ????????? ????????? seed??? ????????? ?????? seeds??? key??? ????????? pool?????? ?????? chromosome ??????
                    chromosome = chromosome_pool.get(seeds[:-1])
                    if chromosome:
                        # pool??? seeds[:-1]??? ???????????? ??????: ????????? chromosome??? ????????? ????????? seed 1?????? ???????????? mutation ??????
                        chromosome = ga_operator.mutation(chromosome, seeds[-1], copy_chromosome=True)
                        del chromosome_pool[seeds[:-1]]
                    else:
                        # pool??? seeds[:-1]??? ???????????? ?????? ??????: ???????????? ?????? seeds??? ????????? ????????? chromosome ??????
                        chromosome = ga_operator.build(seeds)

                else:
                    raise ValueError()

                chromosome_pool[seeds] = chromosome

                episode_reward, steps = ga_operator.evaluate(chromosome)
                worker_to_master_queue.put(
                    MessageFromWorker(
                        seeds=seeds, episode_reward=episode_reward, steps=steps
                    )
                )
                idx_lst.append(idx)
                # print("[GA_WORKER_ID: {0}] SIZE_CHROMOSOME_POOL: {1}, Processed Seeds: {2}".format(
                #     agent.ga_worker_id, len(chromosome_pool), idx_lst
                # ))

            # The pool is updated for every generation.
            # Every new generation is created from the current generation winners.
            # So, there is only a tiny chance that old models (chromosomes) can be reused from the pool.


class GAOperator:
    def __init__(self, env, observation_shape, num_outputs, params, device):
        self.env = env
        self.observation_shape = observation_shape
        self.num_outputs = num_outputs
        self.params = params
        self.device = device

    def mutation(self, chromosome, seed, copy_chromosome=True):
        new_chromosome = copy.deepcopy(chromosome) if copy_chromosome else chromosome
        np.random.seed(seed)
        for parameter in new_chromosome.parameters():
            noise = np.random.normal(size=parameter.data.size())
            noise_v = torch.FloatTensor(noise).to(self.device)
            parameter.data += self.params.NOISE_STANDARD_DEVIATION * noise_v
        return new_chromosome

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

    def build(self, seeds):
        # The first seed is passed to PyTorch to influence the model initialization
        torch.manual_seed(seeds[0])
        chromosome = SimpleModel(
            worker_id=-1,
            observation_shape=self.observation_shape,
            num_outputs=self.num_outputs,
            params=self.params,
            device=self.device
        ).to(self.device)

        # Subsequent seeds are used to apply model mutations
        for seed in seeds[1:]:
            chromosome = self.mutation(chromosome, seed, copy_chromosome=False)

        return chromosome
