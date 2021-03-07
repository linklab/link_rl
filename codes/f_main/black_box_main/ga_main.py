from codes.f_main.general_main.a_common_main import *


def train_main():
    env = rl_utils.get_single_environment(params=params)
    input_shape, num_outputs, action_min, action_max = get_environment_input_output_info(env)

    agent = rl_utils.get_rl_agent(
        input_shape, num_outputs, action_min, action_max, worker_id=-1, params=params, device=device
    )

    agent.initialize(env)
    agent.sort_population_and_set_elite()

    if params.WANDB:
        set_wandb(agent)

    early_stopping = get_early_stopping(agent)

    generation_idx = 0

    while True:
        selected_episode_rewards = [p[1] for p in agent.population[:params.COUNT_FROM_PARENTS]]
        selected_episode_reward_mean = np.mean(selected_episode_rewards)
        selected_episode_reward_max = np.max(selected_episode_rewards)
        selected_episode_reward_std = np.std(selected_episode_rewards)

        print("[GENERATION {0}] episode_reward_mean={1:.2f}, episode_reward_max={2:.2f}, episode_reward_std={3:.2f}".format(
            generation_idx + 1, selected_episode_reward_mean, selected_episode_reward_max, selected_episode_reward_std
        ))

        if params.WANDB:
            train_info_dict = {
                "generation": generation_idx + 1,
                "episode_reward_mean": selected_episode_reward_mean,
                "episode_reward_max": selected_episode_reward_max,
                "episode_reward_std": selected_episode_reward_std
            }
            wandb.log(train_info_dict)

        solved = early_stopping.evaluate(
            evaluation_value=selected_episode_reward_mean,
            episode_done_step=generation_idx
        )

        if solved:
            agent.solved = True
            print("Solved in %d generations" % generation_idx)
            break
        else:
            agent.selection()
            agent.mutation()
            agent.sort_population_and_set_elite()
            generation_idx += 1


if __name__ == "__main__":
    train_main()