# https://github.com/openai/gym/blob/master/gym/envs/classic_control/pendulum.py
# https://mspries.github.io/jimmy_pendulum.html
#!/usr/bin/env python3
from codes.f_main.general_main.common_main import *


def train_main(params, train_env, test_env):
    agent = get_agent(train_env)
    if params.WANDB:
        set_wandb(agent)

    experience_source = ExperienceSourceFirstLast(
        env=train_env, agent=agent, gamma=params.GAMMA, n_step=params.N_STEP
    )

    agent.set_experience_source_to_buffer(experience_source=experience_source)

    step_idx = 0
    early_stopping = get_early_stopping(agent)

    episode = 0
    solved = False

    train_episode_reward_lst_for_stat = deque(maxlen=params.AVG_STEP_SIZE_FOR_TRAIN_LOSS)
    train_episode_reward_lst_for_test = []

    num_tests = get_num_tests()

    loss_dequeue = deque(maxlen=params.AVG_STEP_SIZE_FOR_TRAIN_LOSS)
    actor_objective_dequeue = deque(maxlen=params.AVG_STEP_SIZE_FOR_TRAIN_LOSS)

    with SpeedTracker(params=params) as reward_tracker:
        try:
            while step_idx < params.MAX_GLOBAL_STEP:
                step_idx += params.TRAIN_STEP_FREQ
                last_experience = agent.buffer.populate(params.TRAIN_STEP_FREQ)

                episode_rewards, episode_steps = experience_source.pop_episode_reward_and_done_step_lst()

                if episode_rewards and episode_steps:
                    for current_episode_reward, current_episode_step in zip(episode_rewards, episode_steps):
                        episode += 1

                        solved, train_info_dict = process_episode(
                            train_episode_reward_lst_for_test,
                            train_episode_reward_lst_for_stat,
                            current_episode_reward,
                            agent,
                            reward_tracker,
                            step_idx,
                            episode,
                            test_env,
                            num_tests,
                            early_stopping,
                            current_episode_step,
                            last_experience
                        )

                        mean_loss = np.mean(loss_dequeue) if len(loss_dequeue) > 0 else 0.0
                        mean_actor_objective = np.mean(actor_objective_dequeue) \
                            if len(actor_objective_dequeue) > 0 else 0.0

                        print_performance(
                            params=params,
                            episode_done_step=train_info_dict["step_idx"],
                            done_episode=train_info_dict["episode"],
                            episode_reward=train_info_dict["train episode reward"],
                            mean_episode_reward=train_info_dict[
                                "train mean_{0} episode reward".format(params.AVG_EPISODE_SIZE_FOR_STAT)
                            ],
                            epsilon=train_info_dict["epsilon"] if "epsilon" in train_info_dict else None,
                            elapsed_time=train_info_dict["elapsed_time"],
                            last_info=train_info_dict["last_info"],
                            speed=train_info_dict["speed"],
                            mean_loss=mean_loss,
                            mean_actor_objective=mean_actor_objective,
                            last_action=train_info_dict["last_actions"]
                        )

                        if params.WANDB:
                            train_info_dict["train mean (critic) loss"] = mean_loss
                            train_info_dict["train mean actor objective"] = mean_actor_objective
                            wandb.log(train_info_dict)

                if solved:
                    print("Solved in {0} steps and {1} episodes!".format(step_idx, episode))
                    break
                else:
                    on_policy_agent_train(agent, step_idx, loss_dequeue, actor_objective_dequeue)

            if params.MODEL_SAVE_MODE == ModelSaveMode.FINAL_ONLY:
                last_model_save(agent, step_idx, train_episode_reward_lst_for_stat)
        finally:
            if params.ENVIRONMENT_ID in [EnvironmentName.REAL_DEVICE_RIP, EnvironmentName.REAL_DEVICE_DOUBLE_RIP]:
                print("send 'stop' message into train_env!")
                train_env.stop()


if __name__ == "__main__":
    assert params.RL_ALGORITHM in ON_POLICY_RL_ALGORITHMS

    train_env, test_env = get_train_and_test_envs()

    train_main(params, train_env, test_env)

