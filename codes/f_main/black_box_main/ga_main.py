from codes.f_main.general_main.a_common_main import *


def train_main():
    env = rl_utils.get_single_environment(params=params)
    input_shape, num_outputs, action_min, action_max = get_environment_input_output_info(env)

    agent = rl_utils.get_rl_agent(
        input_shape, num_outputs, action_min, action_max, worker_id=-1, params=params, device=device
    )

    agent.initialize(env)

    early_stopping = get_early_stopping(agent)

    generation_idx = 0

    while True:
        if params.RL_ALGORITHM == RLAlgorithmName.MULTI_GENETIC_ALGORITHM:
            agent.gather_evaluation_results()

        agent.population.sort(key=lambda p: p[1], reverse=True)
        agent.elite = agent.population[0]
        agent.set_best_chromosome()

        selected_episode_rewards = [p[1] for p in agent.population[:params.COUNT_FROM_PARENTS]]
        selected_episode_reward_mean = np.mean(selected_episode_rewards)
        selected_episode_reward_max = np.max(selected_episode_rewards)
        selected_episode_reward_std = np.std(selected_episode_rewards)

        print("[GENERATION {0}] episode_reward_mean={1:.2f}, episode_reward_max={2:.2f}, episode_reward_std={3:.2f}".format(
            generation_idx + 1, selected_episode_reward_mean, selected_episode_reward_max, selected_episode_reward_std
        ))

        solved = early_stopping.evaluate(
            evaluation_value=selected_episode_reward_mean,
            episode_done_step=generation_idx
        )

        if solved:
            agent.solved = True
            print("Solved in %d generations" % generation_idx)
            break
        else:
            agent.next_generation()
            generation_idx += 1


if __name__ == "__main__":
    train_main()