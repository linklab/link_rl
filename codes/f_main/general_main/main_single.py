# https://github.com/openai/gym/blob/master/gym/envs/classic_control/pendulum.py
# https://mspries.github.io/jimmy_pendulum.html
#!/usr/bin/env python3
import time
from collections import deque

import torch
import os, sys
import numpy as np
import wandb
from termcolor import colored

from codes.d_agents.on_policy.on_policy_agent import OnPolicyAgent
from codes.d_agents.off_policy.off_policy_agent import OffPolicyAgent

print("PyTorch Version", torch.__version__)

current_path = os.path.dirname(os.path.realpath(__file__))
PROJECT_HOME = os.path.abspath(os.path.join(current_path, os.pardir, os.pardir, os.pardir))
if PROJECT_HOME not in sys.path:
    sys.path.append(PROJECT_HOME)

from codes.e_utils.experience import ExperienceSourceFirstLast
from codes.e_utils import rl_utils
from codes.e_utils.common_utils import save_model, print_environment_info, print_agent_info, remove_models, \
    agent_model_test, print_performance
from codes.e_utils.experience_tracker import RewardTracker, EarlyStopping
from codes.e_utils.logger import get_logger
from codes.e_utils.names import RLAlgorithmName, EnvironmentName, AgentMode, ModelSaveMode

WANDB_DIR = os.path.join(PROJECT_HOME, "out", "wandb")
if not os.path.exists(WANDB_DIR):
    os.makedirs(WANDB_DIR)

MODEL_SAVE_DIR = os.path.join(PROJECT_HOME, "out", "model_save_files")
if not os.path.exists(MODEL_SAVE_DIR):
    os.makedirs(MODEL_SAVE_DIR)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

my_logger = get_logger("main_single")


def train_main(params, train_env, test_env):
    agent = rl_utils.get_rl_agent(env=train_env, worker_id=0, params=params, device=device)
    print_agent_info(agent, params)

    if params.WANDB:
        configuration = {key: getattr(params, key) for key in dir(params) if not key.startswith("__")}
        wandb.init(
            project=params.wandb_project,
            entity=params.wandb_entity,
            dir=WANDB_DIR,
            config=configuration
        )
        run_name = wandb.run.name
        run_number = run_name.split("-")[-1]
        wandb.run.name = "{0}_{1}_{2}_{3}".format(
            run_number, params.ENVIRONMENT_ID.value, agent.__name__, agent.model.__name__
        )
        wandb.run.save()

    experience_source = ExperienceSourceFirstLast(
        env=train_env, agent=agent, gamma=params.GAMMA, n_step=params.N_STEP, vectorized=True
    )

    agent.set_experience_source_to_buffer(experience_source=experience_source)

    step_idx = 0
    loss_queue = deque(maxlen=100)

    if params.WANDB:
        wandb.watch(agent.model.base)

    early_stopping = EarlyStopping(
        patience=params.STOP_PATIENCE_COUNT,
        evaluation_min_threshold=params.STOP_MEAN_EPISODE_REWARD,
        verbose=True,
        delta=0.0,
        model_save_dir=MODEL_SAVE_DIR,
        model_save_file_prefix=params.ENVIRONMENT_ID.value,
        agent=agent,
        params=params
    )

    if hasattr(agent.train_action_selector, 'epsilon') and hasattr(params, "EPSILON_MIN_STEP"):
        early_stopping.evaluation_min_step_idx = params.EPSILON_MIN_STEP

    episode = 0
    solved = False

    test_mean_episode_reward = 0.0
    train_episode_reward_lst = []

    if params.MODEL_SAVE_MODE == ModelSaveMode.TRAIN:
        num_tests = params.EARLY_STOPPING_TEST_EPISODE_PERIOD
    elif params.MODEL_SAVE_MODE == ModelSaveMode.TEST:
        num_tests = params.TEST_NUM_EPISODES
    else:
        num_tests = 0

    with RewardTracker(params=params) as reward_tracker:
        try:
            while step_idx < params.MAX_GLOBAL_STEP:
                step_idx += params.TRAIN_STEP_FREQ
                last_experience = agent.buffer.populate(params.TRAIN_STEP_FREQ)

                if epsilon_tracker:
                    epsilon_tracker.udpate(step_idx)

                episode_rewards, episode_steps = experience_source.pop_episode_reward_and_done_step_lst()

                if episode_rewards and episode_steps:
                    for current_episode_reward, current_episode_step in zip(episode_rewards, episode_steps):
                        episode += 1
                        train_episode_reward_lst.append(current_episode_reward)

                        epsilon = agent.train_action_selector.epsilon if hasattr(agent.train_action_selector, 'epsilon') else None
                        mean_loss = np.mean(loss_queue) if len(loss_queue) > 0 else 0.0

                        train_mean_episode_reward, speed, elapsed_time = reward_tracker.set_episode_reward(
                            episode_reward=current_episode_reward, episode_done_step=step_idx
                        )

                        print_performance(
                            params=params,
                            episode_done_step=step_idx,
                            done_episode=episode,
                            episode_reward=current_episode_reward,
                            mean_episode_reward=train_mean_episode_reward,
                            epsilon=epsilon,
                            elapsed_time=elapsed_time,
                            last_info=last_experience.info,
                            speed=speed,
                            mean_loss=mean_loss
                        )

                        if episode % params.EARLY_STOPPING_TEST_EPISODE_PERIOD == 0:
                            if params.MODEL_SAVE_MODE in [ModelSaveMode.TRAIN, ModelSaveMode.TEST]:
                                if params.MODEL_SAVE_MODE == ModelSaveMode.TRAIN:
                                    test_mean_episode_reward = np.mean(train_episode_reward_lst)
                                    test_std = np.std(train_episode_reward_lst)
                                    train_episode_reward_lst.clear()
                                    test_env_str = colored("TRAIN ENV", "yellow")
                                else:
                                    test_mean_episode_reward, test_std = agent_model_test(params, test_env, agent)
                                    test_env_str = colored("TEST ENV", "yellow")

                                mean_std_str = colored(
                                    "{0:7.2f}\u00B1{1:.2f}".format(test_mean_episode_reward, test_std), "yellow"
                                )

                                print("[{0:6}/{1}] Ep. {2}: * MODEL SAVE & TRAIN STOP TEST for {3} *, "
                                      "EPISODE REWARD ({4} EPISODES): {5}".format(
                                    step_idx, params.MAX_GLOBAL_STEP, episode, test_env_str, num_tests, mean_std_str
                                ), end="")

                                solved = early_stopping.evaluate(
                                    evaluation_value=test_mean_episode_reward,
                                    episode_done_step=step_idx
                                )
                            elif params.MODEL_SAVE_MODE == ModelSaveMode.FINAL_ONLY:
                                test_mean_episode_reward = 0.0
                                solved = False
                            else:
                                raise ValueError()

                        if params.WANDB:
                            wandb_info = {
                                "train episode reward": train_mean_episode_reward,
                                "train mean_{0} episode reward".format(params.AVG_EPISODE_SIZE_FOR_STAT):
                                    train_mean_episode_reward,
                                "test mean_{0} episode reward".format(num_tests): test_mean_episode_reward,
                                "steps/episode": current_episode_step,
                                "speed": speed,
                                "step_idx": step_idx,
                                "episode": episode,
                                'actions': last_experience.action,
                            }
                            if epsilon:
                                wandb_info["epsilon"] = epsilon
                            wandb.log(wandb_info)

                if solved:
                    print("Solved in {0} steps and {1} episodes!".format(step_idx, episode))
                    break

                if isinstance(agent, OnPolicyAgent):
                    if params.RL_ALGORITHM in [RLAlgorithmName.CONTINUOUS_PPO_V0, RLAlgorithmName.DISCRETE_PPO_V0]:
                        if len(agent.buffer) < params.PPO_TRAJECTORY_SIZE:
                            continue
                    else:
                        if len(agent.buffer) < params.BATCH_SIZE:
                            continue
                    _, last_loss, _ = agent.train(step_idx=step_idx)
                    agent.buffer.clear()
                elif isinstance(agent, OffPolicyAgent):
                    if len(agent.buffer) < params.MIN_REPLAY_SIZE_FOR_TRAIN:
                        continue
                    _, last_loss, _ = agent.train(step_idx=step_idx)
                else:
                    raise ValueError()

                loss_queue.append(last_loss)

                if hasattr(params, "PER_RANK_BASED") and getattr(params, "PER_RANK_BASED"):
                    if step_idx % 100 < params.TRAIN_STEP_FREQ:
                        agent.buffer.rebalance()

            if params.MODEL_SAVE_MODE == ModelSaveMode.FINAL_ONLY:
                remove_models(
                    MODEL_SAVE_DIR, params.ENVIRONMENT_ID.value, agent
                )
                save_model(
                    MODEL_SAVE_DIR, params.ENVIRONMENT_ID.value, agent, step_idx, train_mean_episode_reward
                )
        finally:
            if params.ENVIRONMENT_ID in [EnvironmentName.REAL_DEVICE_RIP, EnvironmentName.REAL_DEVICE_DOUBLE_RIP]:
                train_env.stop()


if __name__ == "__main__":
    from codes.a_config.parameters import PARAMETERS as parameters
    params = parameters

    train_env = rl_utils.get_environment(params=params)
    print_environment_info(train_env, params)

    if params.MODEL_SAVE_MODE == ModelSaveMode.TEST:
        test_env = rl_utils.get_single_environment(params=params)
    else:
        test_env = None

    train_main(params, train_env, test_env)

