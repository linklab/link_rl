# https://github.com/openai/gym/blob/master/gym/envs/classic_control/pendulum.py
# https://mspries.github.io/jimmy_pendulum.html
#!/usr/bin/env python3
import collections
import time
from collections import deque

import torch
import torch.multiprocessing as mp
import os, sys
import numpy as np
import wandb

from codes.d_agents.off_policy.off_policy_agent import OffPolicyAgent
from codes.d_agents.on_policy.on_policy_agent import OnPolicyAgent

print("PyTorch Version", torch.__version__)

current_path = os.path.dirname(os.path.realpath(__file__))
PROJECT_HOME = os.path.abspath(os.path.join(current_path, os.pardir, os.pardir, os.pardir))
if PROJECT_HOME not in sys.path:
    sys.path.append(PROJECT_HOME)

from codes.a_config.parameters import PARAMETERS as params

from codes.e_utils import rl_utils
from codes.e_utils.common_utils import save_model, print_environment_info, remove_models, agent_model_test
from codes.e_utils.experience_tracker import RewardTracker, EarlyStopping
from codes.e_utils.logger import get_logger
from codes.e_utils.names import DeepLearningModelName, RLAlgorithmName, EnvironmentName, ModelSaveMode
from codes.e_utils.experience import ExperienceSourceFirstLast

WANDB_DIR = os.path.join(PROJECT_HOME, "out", "wandb")
if not os.path.exists(WANDB_DIR):
    os.makedirs(WANDB_DIR)

MODEL_SAVE_DIR = os.path.join(PROJECT_HOME, "out", "model_save_files")
if not os.path.exists(MODEL_SAVE_DIR):
    os.makedirs(MODEL_SAVE_DIR)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

my_logger = get_logger("main")

WandbInfo = collections.namedtuple('WandbInfo', field_names='wandb_info')


def play_func(exp_queue, agent, epsilon_tracker):
    train_env = rl_utils.get_environment(params=params)
    print_environment_info(train_env, params)

    if params.MODEL_SAVE_MODE == ModelSaveMode.TEST:
        test_env = rl_utils.get_single_environment(params=params)
    else:
        test_env = None

    if params.DEEP_LEARNING_MODEL in [
        DeepLearningModelName.DETERMINISTIC_CONTINUOUS_ACTOR_CRITIC_GRU,
        DeepLearningModelName.DETERMINISTIC_CONTINUOUS_ACTOR_CRITIC_GRU_ATTENTION
    ]:
        step_length = params.RNN_STEP_LENGTH
    else:
        step_length = -1

    experience_source = ExperienceSourceFirstLast(
        env=train_env, agent=agent, gamma=params.GAMMA, n_step=params.N_STEP, vectorized=True
    )

    exp_source_iter = iter(experience_source)
    step_idx = 0

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

    with RewardTracker(params=params) as reward_tracker:
        try:
            while step_idx < params.MAX_GLOBAL_STEP:
                # 1 스텝 진행하고 exp를 exp_queue에 넣음
                step_idx += 1
                exp = next(exp_source_iter)
                exp_queue.put(exp)

                if epsilon_tracker:
                    epsilon_tracker.udpate(step_idx)

                episode_rewards, episode_steps = experience_source.pop_episode_reward_and_done_step_lst()

                if episode_rewards and episode_steps:
                    for current_episode_reward, current_episode_step in zip(episode_rewards, episode_steps):
                        episode += 1

                        epsilon = agent.train_action_selector.epsilon if hasattr(agent.train_action_selector, 'epsilon') else None

                        train_mean_episode_reward, speed = reward_tracker.set_episode_reward(
                            episode_reward=current_episode_reward, episode_done_step=step_idx,
                            epsilon=epsilon, last_info=exp.info
                        )
                        if params.MODEL_SAVE_MODE == ModelSaveMode.TRAIN:
                            if episode % params.EARLY_STOPPING_TEST_EPISODE_PERIOD == 0:
                                test_mean_episode_reward = 0.0
                                print("[{0:6}/{1}] Ep. {2}: * MODEL SAVE TEST * TRAIN EPISODE REWARD: {3:7.2f} ".format(
                                    step_idx, params.MAX_GLOBAL_STEP, episode, train_mean_episode_reward
                                ), end="")
                                solved = early_stopping.evaluate(
                                    evaluation_value=train_mean_episode_reward,
                                    episode_done_step=step_idx
                                )
                        elif params.MODEL_SAVE_MODE == ModelSaveMode.TEST:
                            if episode % params.EARLY_STOPPING_TEST_EPISODE_PERIOD == 0:
                                test_mean_episode_reward = agent_model_test(params, test_env, agent)
                                print("[{0:6}/{1}] Ep. {2}: * MODEL SAVE TEST * TEST EPISODE REWARD: {3:7.2f} ".format(
                                    step_idx, params.MAX_GLOBAL_STEP, episode, test_mean_episode_reward
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
                            wandb_info_dict = {
                                "train episode reward": train_mean_episode_reward,
                                "test episode reward": test_mean_episode_reward,
                                "steps/episode": current_episode_step,
                                "speed": speed,
                                "step_idx": step_idx,
                                "episode": episode
                            }
                            if epsilon:
                                wandb_info_dict["epsilon"] = epsilon

                            wandb_info = WandbInfo(wandb_info=wandb_info_dict)
                            exp_queue.put(wandb_info)

                if solved:
                    print("Solved in {0} steps and {1} episodes!".format(step_idx, episode))
                    break

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

    exp_queue.put(None)


def main():
    mp.set_start_method('spawn')
    os.environ['OMP_NUM_THREADS'] = "1"

    env = rl_utils.get_single_environment(params=params)
    agent, epsilon_tracker = rl_utils.get_rl_agent(env=env, worker_id=0, params=params, device=device)
    agent.model.share_memory()

    exp_queue = mp.Queue(maxsize=params.TRAIN_STEP_FREQ * 2)
    play_proc = mp.Process(target=play_func, args=(exp_queue, agent, epsilon_tracker))
    play_proc.start()

    time.sleep(0.5)

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

        wandb.watch(agent.model.base)

    loss_dequeue = deque(maxlen=params.AVG_STEP_SIZE_FOR_TRAIN_LOSS)
    step_idx = 0

    while play_proc.is_alive():
        step_idx += params.TRAIN_STEP_FREQ
        for _ in range(params.TRAIN_STEP_FREQ):
            exp = exp_queue.get()
            if isinstance(exp, WandbInfo):
                mean_loss = np.mean(loss_dequeue) if len(loss_dequeue) > 0 else 0.0
                exp.wandb_info["train (critic) mean loss"] = mean_loss
                wandb.log(exp.wandb_info)
                continue

            if exp is None:
                play_proc.join()
                break
            agent.buffer._add(exp)

        if isinstance(agent, OnPolicyAgent):
            if params.RL_ALGORITHM in [RLAlgorithmName.CONTINUOUS_PPO_V0, RLAlgorithmName.DISCRETE_PPO_V0]:
                if len(agent.buffer) < params.PPO_TRAJECTORY_SIZE:
                    continue
            else:
                if len(agent.buffer) < params.BATCH_SIZE:
                    continue
            _, last_loss, _ = agent.train(step_idx=step_idx)
            loss_dequeue.append(last_loss)
            # On-policy는 현재의 정책을 통해 산출된 경험정보만을 활용하여 NN을 업데이트해야 함.
            # 따라서, 현재 학습에 사용된 Buffer는 깨끗하게 지워야 함.
            agent.buffer.clear()
        elif isinstance(agent, OffPolicyAgent):
            if len(agent.buffer) < params.MIN_REPLAY_SIZE_FOR_TRAIN:
                continue

            if params.ENVIRONMENT_ID == EnvironmentName.QUANSER_SERVO_2:
                # ===============================20 train for one step================================
                last_loss = 0.0
                for i in range(20):
                    _, last_loss, _ = agent.train(step_idx=step_idx)
                #=====================================================================================
            else:
                _, last_loss, _ = agent.train(step_idx=step_idx)
            loss_dequeue.append(last_loss)
        else:
            raise ValueError()

        if hasattr(params, "PER_RANK_BASED") and getattr(params, "PER_RANK_BASED"):
            if step_idx % 100 < params.TRAIN_STEP_FREQ:
                agent.buffer.rebalance()


if __name__ == "__main__":
    main()