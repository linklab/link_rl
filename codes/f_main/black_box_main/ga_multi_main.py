from codes.d_agents.black_box.cma_es.ga_agent import AgentGA
from codes.f_main.general_main.a_common_main import *
import torch.multiprocessing as mp
import collections

OutputItem = collections.namedtuple(
    'OutputItem', field_names=['seeds', 'episode_reward', 'steps']
)


def worker_func(input_queue, output_queue):
    env = rl_utils.get_single_environment(params=params)
    input_shape, num_outputs, action_min, action_max = get_environment_input_output_info(env)

    agent = rl_utils.get_rl_agent(
        input_shape, num_outputs, action_min, action_max, worker_id=-1, params=params, device=device
    )

    agent.initialize(env)

    cache = {}

    while True:
        parents = input_queue.get()
        if parents is None:
            break

        new_cache = {}
        for chromosome_seeds in parents:
            if len(chromosome_seeds) > 1:
                chromosome = cache.get(chromosome_seeds[:-1])
                if chromosome is not None:
                    chromosome = agent.mutate(chromosome, chromosome_seeds[-1])
                else:
                    chromosome = agent.build(chromosome_seeds)
            else:
                chromosome = agent.build(chromosome_seeds)

            new_cache[chromosome_seeds] = chromosome
            episode_reward, steps = agent.evaluate(chromosome)
            output_queue.put(OutputItem(seeds=chromosome_seeds, episode_reward=episode_reward, steps=steps))
        cache = new_cache


def train_main():
    mp.set_start_method('spawn')

    input_queues = []
    output_queue = mp.Queue(maxsize=params.WORKERS_COUNT)

    for _ in range(params.WORKERS_COUNT):
        input_queue = mp.Queue(maxsize=1)
        input_queues.append(input_queue)
        w = mp.Process(target=worker_func, args=(input_queue, output_queue))
        w.start()
        seeds = [(np.random.randint(params.MAX_SEED),) for _ in range(params.SEEDS_PER_WORKER)]
        input_queue.put(seeds)

    early_stopping = get_early_stopping(None)
    generation_idx = 0
    elite = None

    while True:
        batch_steps = 0
        population = []

        while len(population) < params.SEEDS_PER_WORKER * params.WORKERS_COUNT:
            out_item = output_queue.get()
            population.append((out_item.seeds, out_item.episode_reward))
            batch_steps += out_item.steps

        if elite is not None:
            population.append(elite)

        population.sort(key=lambda p: p[1], reverse=True)
        selected_episode_rewards = [p[1] for p in population[:params.PARENTS_COUNT]]
        selected_episode_reward_mean = np.mean(selected_episode_rewards)
        selected_episode_reward_max = np.max(selected_episode_rewards)
        selected_episode_reward_std = np.std(selected_episode_rewards)

        print("{0}: episode_reward_mean={1:.2f}, episode_reward_max={2:.2f}, episode_reward_std={3:.2f}".format(
            generation_idx, selected_episode_reward_mean, selected_episode_reward_max, selected_episode_reward_std
        ))

        solved = early_stopping.evaluate(
            evaluation_value=selected_episode_reward_mean,
            episode_done_step=generation_idx
        )

        if solved:
            print("Solved in %d generations" % generation_idx)
            break
        else:
            elite = population[0]
            for worker_queue in input_queues:
                seeds = []
                for _ in range(params.SEEDS_PER_WORKER):
                    parent = np.random.randint(params.PARENTS_COUNT)
                    next_seed = np.random.randint(params.MAX_SEED)
                    s = list(population[parent][0]) + [next_seed]
                    seeds.append(tuple(s))
                worker_queue.put(seeds)
            generation_idx += 1


if __name__ == "__main__":
    train_main()