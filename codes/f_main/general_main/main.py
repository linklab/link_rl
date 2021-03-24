# https://github.com/openai/gym/blob/master/gym/envs/classic_control/pendulum.py
# https://mspries.github.io/jimmy_pendulum.html
#!/usr/bin/env python3
import os, sys

current_path = os.path.dirname(os.path.realpath(__file__))
PROJECT_HOME = os.path.abspath(os.path.join(current_path, os.pardir, os.pardir, os.pardir))
if PROJECT_HOME not in sys.path:
    sys.path.append(PROJECT_HOME)

from codes.f_main.general_main.a_common_main import *


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
    train_episode_reward_lst_for_test = deque(maxlen=params.TEST_NUM_EPISODES)

    num_tests = get_num_tests()

    loss_dequeue = deque(maxlen=params.AVG_STEP_SIZE_FOR_TRAIN_LOSS)
    actor_objective_dequeue = deque(maxlen=params.AVG_STEP_SIZE_FOR_TRAIN_LOSS)

    with SpeedTracker(params=params) as reward_tracker:
        try:
            while step_idx < params.MAX_GLOBAL_STEP:
                step_idx += params.TRAIN_STEP_FREQ
                exp = agent.buffer.populate(params.TRAIN_STEP_FREQ)

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
                            exp
                        )

                        mean_loss = np.mean(loss_dequeue) if len(loss_dequeue) > 0 else 0.0
                        mean_actor_objective = np.mean(actor_objective_dequeue) \
                            if len(actor_objective_dequeue) > 0 else 0.0

                        print_performance(
                            params=params,
                            episode_done_step=train_info_dict["step_idx"],
                            done_episode=train_info_dict["episode"],
                            episode_reward=train_info_dict["### EVERY TRAIN EPISODE REWARDS ###"],
                            mean_episode_reward=train_info_dict[
                                "train mean ({0} episode rewards)".format(params.AVG_EPISODE_SIZE_FOR_STAT)
                            ],
                            epsilon=train_info_dict["epsilon"] if "epsilon" in train_info_dict else None,
                            elapsed_time=train_info_dict["elapsed_time"],
                            last_info=train_info_dict["last_info"],
                            speed=train_info_dict["speed"],
                            mean_loss=mean_loss,
                            mean_actor_objective=mean_actor_objective,
                            last_action=train_info_dict["last_actions"],
                            evaluation_msg=train_info_dict["evaluation_msg"]
                        )

                        if params.WANDB:
                            train_info_dict["train mean (critic) loss"] = mean_loss
                            train_info_dict["train mean actor objective"] = mean_actor_objective
                            del train_info_dict["evaluation_msg"]
                            del train_info_dict["solved"]
                            wandb.log(train_info_dict)

                    if params.TRAIN_ONLY_AFTER_EPISODE:
                        # num_trains = int((step_idx - last_train_step_idx)/(2 * params.TRAIN_STEP_FREQ))
                        # print(step_idx, last_train_step_idx, num_trains)
                        for _ in range(params.NUM_TRAIN_ONLY_AFTER_EPISODE):
                            train(agent, step_idx, loss_dequeue, actor_objective_dequeue)

                        if params.RL_ALGORITHM in [RLAlgorithmName.DDPG_V0]:
                            if params.TYPE_OF_TARGET_UPDATE == "hard_update":
                                agent.target_agent.alpha_sync(alpha=0.0)
                            elif params.TYPE_OF_TARGET_UPDATE == "soft_update":
                                agent.target_agent.alpha_sync(alpha=0.75)  # 0.75: 새로운 파라미터는 0.25만 반영
                            else:
                                raise ValueError()

                if solved:
                    print("Solved in {0} steps and {1} episodes!".format(step_idx, episode))
                    break

                if not params.TRAIN_ONLY_AFTER_EPISODE:
                    train(agent, step_idx, loss_dequeue, actor_objective_dequeue)

            if params.MODEL_SAVE_MODE == ModelSaveMode.FINAL_ONLY:
                last_model_save(agent, step_idx, train_episode_reward_lst_for_stat)
        finally:
            if params.ENVIRONMENT_ID in [EnvironmentName.REAL_DEVICE_RIP, EnvironmentName.REAL_DEVICE_DOUBLE_RIP]:
                print("send 'stop' message into train_env!")
                train_env.stop()


def train(agent, step_idx, loss_dequeue, actor_objective_dequeue):
    if params.RL_ALGORITHM in ON_POLICY_RL_ALGORITHMS:
        if params.RL_ALGORITHM in [RLAlgorithmName.CONTINUOUS_PPO_V0, RLAlgorithmName.DISCRETE_PPO_V0]:
            if len(agent.buffer) < params.PPO_TRAJECTORY_SIZE:
                return
        else:
            if len(agent.buffer) < params.BATCH_SIZE:
                return

        _, last_loss, actor_objective = agent.train(step_idx=step_idx)

        loss_dequeue.append(last_loss)

        if actor_objective:
            actor_objective_dequeue.append(actor_objective)

        # On-policy는 현재의 정책을 통해 산출된 경험정보만을 활용하여 NN을 업데이트해야 함.
        # 따라서, 현재 학습에 사용된 Buffer는 깨끗하게 지워야 함.
        agent.buffer.clear()
    else:
        if len(agent.buffer) < params.MIN_REPLAY_SIZE_FOR_TRAIN:
            return

        _, last_loss, actor_objective = agent.train(step_idx=step_idx)

        loss_dequeue.append(last_loss)

        if actor_objective:
            actor_objective_dequeue.append(actor_objective)

        if hasattr(params, "PER_RANK_BASED") and getattr(params, "PER_RANK_BASED"):
            if step_idx % 100 < params.TRAIN_STEP_FREQ:
                agent.buffer.rebalance()


if __name__ == "__main__":
    #assert params.RL_ALGORITHM in ON_POLICY_RL_ALGORITHMS

    if params.TRAIN_ONLY_AFTER_EPISODE:
        assert params.NUM_ENVIRONMENTS == 1

    train_env, test_env = get_train_and_test_envs()

    train_main(params, train_env, test_env)