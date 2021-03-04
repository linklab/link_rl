import collections
import numpy as np
import copy
import torch
import torch.multiprocessing as mp

from codes.c_models.discrete_action.simple_model import SimpleModel
from codes.d_agents.a0_base_agent import BaseAgent
from codes.e_utils import rl_utils

MessageFromWorker = collections.namedtuple(
    'MessageFromWorker', field_names=['seeds', 'episode_reward', 'steps', 'best_chromosome']
)

MessageFromMaster = collections.namedtuple(
    'MessageFromMaster', field_names=['seeds_lst', 'best_seeds']
)

# Deep Neuroevolution: Genetic Algorithms are a Competitive
# Alternative for Training Deep Neural Networks for Reinforcement Learning

class AgentMultiGA(BaseAgent):
    def __init__(self, worker_id, input_shape, num_outputs, params, device):
        super(AgentMultiGA, self).__init__(worker_id, params, device)
        self.__name__ = "AgentMultiGA"

        self.model = SimpleModel(
            worker_id=worker_id,
            input_shape=input_shape,
            num_outputs=num_outputs,
            params=params,
            device=device
        ).to(self.device)

        self.env = None
        self.population = None
        self.elite = None
        self.best_chromosome = None
        self.global_steps = 0

        mp.set_start_method('spawn')

        self.master_to_worker_queue_lst = []
        self.worker_to_master_queue = mp.Queue(maxsize=params.WORKERS_COUNT)

        self.solved = False

    def initialize(self, env):
        self.env = env

        for ga_worker_idx in range(self.params.WORKERS_COUNT):
            master_to_worker_queue = mp.Queue(maxsize=1)
            self.master_to_worker_queue_lst.append(master_to_worker_queue)
            worker = mp.Process(
                target=self.worker_func,
                args=(ga_worker_idx, master_to_worker_queue, self.worker_to_master_queue, self.params, self.device)
            )
            worker.start()

            # 최초 seeds 및 seeds_lst 구성
            seeds_lst = []
            # params.NUM_SEEDS_PER_WORKER: 300
            for _ in range(self.params.NUM_SEEDS_PER_WORKER):
                seeds = (np.random.randint(self.params.MAX_SEED),)
                seeds_lst.append(seeds)

            master_to_worker_queue.put(MessageFromMaster(seeds_lst=seeds_lst, best_seeds=None))

    def gather_evaluation_results(self):
        self.population = []

        if self.elite:
            self.population.append(self.elite)

        while len(self.population) < self.params.POPULATION_SIZE:
            message = self.worker_to_master_queue.get()

            self.population.append((message.seeds, message.episode_reward))
            self.global_steps += message.steps

            if message.best_chromosome:
                self.best_chromosome = message.best_chromosome

    def next_generation(self):
        best_seeds = self.elite[0]

        # 각 worker들에게 새로운 seeds_lst 전달
        for master_to_worker_queue in self.master_to_worker_queue_lst:
            if self.solved:
                master_to_worker_queue.put("Solved")
            else:
                seeds_lst = []
                for _ in range(self.params.NUM_SEEDS_PER_WORKER):
                    # population내에서 episord_reward 기준 10위 이내의 chromosome을 임의로 선택
                    parent = np.random.randint(self.params.COUNT_FROM_PARENTS)
                    next_seed = np.random.randint(self.params.MAX_SEED)

                    # 그 선택된 parent chromosome을 만들 때 사용한 seeds에 새로운 seed를 더하여 새로운 seeds 구성
                    # (1099612850, 3655502209) --> (1099612850, 3655502209, 1087985398)
                    seeds = list(self.population[parent][0]) + [next_seed]
                    seeds_lst.append(tuple(seeds))
                master_to_worker_queue.put(MessageFromMaster(seeds_lst=seeds_lst, best_seeds=best_seeds))

    def set_best_chromosome(self):
        if self.best_chromosome:
            self.model.load_state_dict(self.best_chromosome.state_dict())

    def __call__(self, states, agent_states=None):
        states = states[0]
        action_prob = self.model(states)
        acts = action_prob.max(dim=1)[1]

        return acts.data.cpu().numpy(), None

    @staticmethod
    def worker_func(ga_worker_id, master_to_worker_queue, worker_to_master_queue, params, device):
        env = rl_utils.get_single_environment(params=params)
        input_shape, num_outputs, action_min, action_max = rl_utils.get_environment_input_output_info(env)

        agent = WorkerAgentMultiGA(
            ga_worker_id=ga_worker_id, env=env, input_shape=input_shape, num_outputs=num_outputs, params=params, device=device
        )

        chromosome_pool = {}

        while True:
            message = master_to_worker_queue.get()

            if type(message) is str and message == "Solved":
                break

            seeds_lst = message.seeds_lst
            best_seeds = message.best_seeds

            if best_seeds in chromosome_pool:
                best_chromosome = chromosome_pool[best_seeds]
            else:
                best_chromosome = None

            # seeds_lst에는 params.NUM_SEEDS_PER_WORKER (300)개수의 seeds 존재
            # seeds는 일련의 seed로 구성된 tuple
            assert len(seeds_lst) == params.NUM_SEEDS_PER_WORKER

            # Pool of models (chromosomes): minimize the amount of time spent recreating the parameters from the same seeds.
            # Master로 부터 전달받은 seeds에 대해
            new_chromosome_pool = {}

            idx_lst = []
            for idx, seeds in enumerate(seeds_lst):
                if len(seeds) == 1:
                    # seeds에 1개의 seed만 있는 경우 --> 최초 chromosome 생성
                    chromosome = agent.build(seeds)

                elif len(seeds) > 1:
                    # seeds에 2개 이상의 seed가 있는 경우
                    # 새롭게 추가된 seed를 제외한 기존 seeds를 key로 정하여 pool에서 해당 chromosome 검색
                    chromosome = chromosome_pool.get(seeds[:-1])
                    if chromosome:
                        # pool에 이미 존재하는 seeds[:-1]인 경우: 연관된 chromosome에 새롭게 추가된 seed 1개에 대해서만 mutate 수행
                        chromosome = agent.mutate(chromosome, seeds[-1], copy_chromosome=True)
                    else:
                        # pool에 존재하지 않는 seeds[:-1]인 경우: 전달받은 전체 seeds에 대해서 새로운 chromosome 생성
                        chromosome = agent.build(seeds)

                else:
                    raise ValueError()

                new_chromosome_pool[seeds] = chromosome
                episode_reward, steps = agent.evaluate(chromosome)  # 총 evaluate 횟수: params.NUM_SEEDS_PER_WORKER
                worker_to_master_queue.put(
                    MessageFromWorker(
                        seeds=seeds, episode_reward=episode_reward, steps=steps, best_chromosome=best_chromosome
                    )
                )
                idx_lst.append(idx)
                print("[GA_WORKER_ID: {0}] SIZE_CHROMOSOME_POOL: {1}, Processed Seeds: {2}".format(
                    agent.ga_worker_id, len(chromosome_pool), idx_lst
                ))

            # The pool is cleared for every generation.
            # Every new generation is created from the current generation winners.
            # So, there is only a tiny chance that old models (chromosomes) can be reused from the pool.
            chromosome_pool = new_chromosome_pool


class WorkerAgentMultiGA():
    def __init__(self, ga_worker_id, env, input_shape, num_outputs, params, device):
        self.ga_worker_id = ga_worker_id
        self.env = env
        self.input_shape = input_shape
        self.num_outputs = num_outputs
        self.params = params
        self.device = device

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
            noise_v = torch.FloatTensor(noise).to(self.device)
            parameter.data += self.params.NOISE_STANDARD_DEVIATION * noise_v
        return new_chromosome

    def build(self, seeds):
        # The first seed is passed to PyTorch to influence the model initialization
        torch.manual_seed(seeds[0])
        chromosome = SimpleModel(
            worker_id=-1,
            input_shape=self.input_shape,
            num_outputs=self.num_outputs,
            params=self.params,
            device=self.device
        ).to(self.device)

        # Subsequent seeds are used to apply model mutations
        for seed in seeds[1:]:
            chromosome = self.mutate(chromosome, seed, copy_chromosome=False)

        return chromosome
